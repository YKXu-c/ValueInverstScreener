import requests
import time
import json
import os
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from collections import deque

# ==================== 配置日志 ====================
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = f"{log_dir}/graham_screener_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== 数据类定义 ====================
@dataclass
class Stock:
    """股票数据类"""
    symbol: str  # 6位数字代码
    name: str
    exchange: str  # 交易所
    price: float = 0.0
    pe: float = 0.0
    pb: float = 0.0
    dvr: float = 0.0  # 股息率
    total_debt: float = 0.0  # 总负债（万元）
    current_debt: float = 0.0  # 流动负债（万元）
    current_assets: float = 0.0  # 流动资产（万元）
    current_ratio: float = 0.0  # 流动比率
    net_assets: float = 0.0  # 净资产（万元）
    total_assets: float = 0.0  # 总资产（万元）
    eps: float = 0.0  # 每股收益（元）
    total_shares: float = 0.0  # 总股本（万股）
    market_cap: float = 0.0  # 市值（亿元）
    dividend_cash_paid: float = 0.0  # 分配股利现金（万元）
    interest_expense: float = 0.0  # 利息支出（万元）
    
    # 筛选结果记录
    step_results: List[bool] = field(default_factory=list)
    step_summary: str = ""
    
    # 历史数据
    historical_eps: List[float] = field(default_factory=list)  # 历史每股收益（最近5年）
    historical_prices: Dict[str, float] = field(default_factory=dict)  # 历史价格 {年月: 价格}
    historical_pe: List[float] = field(default_factory=list)  # 历史市盈率（最近5年）
    
    def calculate_metrics(self):
        """计算各项指标"""
        # 计算市盈率
        if self.eps > 0:
            self.pe = self.price / self.eps
        else:
            self.pe = 999  # 如果没有EPS数据，设置一个高值
        
        # 计算市净率
        if self.total_shares > 0 and self.net_assets > 0:
            # 注意：net_assets是万元，total_shares是万股
            # 每股净资产 = 净资产(万元) / 总股本(万股)
            book_value_per_share = self.net_assets / self.total_shares  # 元/股
            if book_value_per_share > 0:
                self.pb = self.price / book_value_per_share
        else:
            self.pb = 999
        
        # 计算股息率
        if self.price > 0 and self.total_shares > 0 and self.dividend_cash_paid > 0:
            # 每股分红 = 分红现金(万元) / 总股本(万股)
            dividend_per_share = self.dividend_cash_paid / self.total_shares  # 元/股
            self.dvr = dividend_per_share / self.price
        
        # 计算流动比率
        if self.current_debt > 0:
            self.current_ratio = self.current_assets / self.current_debt
        
        # 计算市值（亿元）
        if self.price > 0 and self.total_shares > 0:
            # 市值 = 股价(元) × 总股本(万股) / 10000
            self.market_cap = self.price * self.total_shares / 10000  # 亿元

# ==================== 速率限制器 ====================
class RateLimiter:
    """请求速率限制器"""
    
    def __init__(self, max_requests_per_minute: int = 280):
        self.max_requests_per_minute = max_requests_per_minute
        self.request_times = deque()
        self.min_interval = 60.0 / max_requests_per_minute  # 最小请求间隔
        self.last_request_time = 0
        
    def wait_if_needed(self):
        """如果需要等待，则等待适当时间"""
        current_time = time.time()
        
        # 移除超过1分钟的请求时间
        while self.request_times and current_time - self.request_times[0] > 60:
            self.request_times.popleft()
        
        # 如果达到限制，等待
        if len(self.request_times) >= self.max_requests_per_minute:
            wait_time = 60 - (current_time - self.request_times[0]) + 0.1
            if wait_time > 0:
                logger.debug(f"达到速率限制，等待 {wait_time:.1f} 秒")
                time.sleep(wait_time)
                current_time = time.time()
                while self.request_times and current_time - self.request_times[0] > 60:
                    self.request_times.popleft()
        
        # 确保最小请求间隔
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            time.sleep(wait_time)
            current_time = time.time()
        
        # 记录这次请求
        self.request_times.append(current_time)
        self.last_request_time = current_time

# ==================== 格雷厄姆筛选器 ====================
class GrahamScreener:
    """格雷厄姆筛选器"""
    
    def __init__(self, licence: str, test_mode: bool = True):
        self.licence = licence
        self.test_mode = test_mode
        self.base_url = "https://api.mairuiapi.com"
        self.rate_limiter = RateLimiter(max_requests_per_minute=280)
        self.good_stocks = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        })
        
        # 配置重试策略
        self.retry_count = 3
        self.retry_delay = 5
        
        # 缓存数据以减少API调用
        self.stock_info_cache = {}
    
    def make_request(self, url: str) -> Optional[Dict]:
        """发送API请求，包含速率限制和重试机制"""
        for attempt in range(self.retry_count):
            try:
                # 应用速率限制
                self.rate_limiter.wait_if_needed()
                
                logger.debug(f"发送请求: {url}")
                response = self.session.get(url, timeout=15)
                
                if response.status_code == 200:
                    # 尝试解析JSON数据
                    try:
                        data = response.json()
                        # API返回的是标准JSON格式，可能是列表或字典
                        logger.debug(f"请求成功，返回数据格式: {type(data)}")
                        return data
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON解析失败: {e}")
                        logger.error(f"响应内容: {response.text[:200]}")
                        return None
                elif response.status_code == 429:
                    wait_time = 30 * (attempt + 1)
                    logger.warning(f"请求过多(429)，等待 {wait_time} 秒后重试")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 403:
                    logger.error(f"许可证无效或访问被拒绝: {url}")
                    return None
                elif response.status_code == 404:
                    logger.warning(f"数据不存在: {url}")
                    return None
                else:
                    logger.warning(f"请求失败 {url}: {response.status_code}")
                    logger.warning(f"响应: {response.text[:200]}")
                    if attempt < self.retry_count - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return None
                    
            except requests.exceptions.Timeout:
                logger.warning(f"请求超时: {url}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
                    continue
                return None
            except requests.exceptions.RequestException as e:
                logger.error(f"请求异常 {url}: {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
                    continue
                return None
        
        return None
    
    def get_stock_list(self) -> List[Dict]:
        """获取股票列表"""
        url = f"{self.base_url}/hslt/list/{self.licence}"
        logger.info("获取股票列表...")
        
        data = self.make_request(url)
        if data and isinstance(data, list):
            logger.info(f"成功获取 {len(data)} 支股票")
            
            # 记录前十组股票信息到日志
            logger.info("前十组股票信息:")
            for i, stock in enumerate(data[:10]):
                logger.info(f"  {i+1:2d}. 代码: {stock.get('dm', ''):<12} 名称: {stock.get('mc', ''):<20} 交易所: {stock.get('jys', '')}")
            
            # 只返回前50支用于测试
            if self.test_mode:
                return data[:50]
            return data
        elif data and isinstance(data, dict):
            # 如果返回的是字典，尝试提取列表
            logger.warning("API返回的是字典格式，尝试提取列表...")
            if 'data' in data:
                stocks = data['data']
                if isinstance(stocks, list):
                    logger.info(f"成功获取 {len(stocks)} 支股票")
                    return stocks[:50] if self.test_mode else stocks
            return []
        
        # 使用模拟数据测试
        logger.warning("API请求失败，使用模拟数据测试")
        return self.get_mock_stock_list()
    
    def get_mock_stock_list(self) -> List[Dict]:
        """获取模拟股票列表用于测试"""
        mock_stocks = [
            {"dm": "000001.SZ", "mc": "平安银行", "jys": "SZ"},
            {"dm": "000002.SZ", "mc": "万科A", "jys": "SZ"},
            {"dm": "000858.SZ", "mc": "五粮液", "jys": "SZ"},
            {"dm": "000333.SZ", "mc": "美的集团", "jys": "SZ"},
            {"dm": "000651.SZ", "mc": "格力电器", "jys": "SZ"},
            {"dm": "600036.SH", "mc": "招商银行", "jys": "SH"},
            {"dm": "600519.SH", "mc": "贵州茅台", "jys": "SH"},
            {"dm": "601318.SH", "mc": "中国平安", "jys": "SH"},
            {"dm": "601398.SH", "mc": "工商银行", "jys": "SH"},
            {"dm": "601939.SH", "mc": "建设银行", "jys": "SH"},
        ]
        return mock_stocks[:5] if self.test_mode else mock_stocks
    
    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        """获取股票基本信息 - 使用完整的股票代码（如000001.SZ）"""
        # 清理股票代码格式
        if '.' not in symbol:
            # 如果没有后缀，加上默认的后缀
            if symbol.startswith('6'):
                symbol = f"{symbol}.SH"
            else:
                symbol = f"{symbol}.SZ"
        
        # 检查缓存
        if symbol in self.stock_info_cache:
            return self.stock_info_cache[symbol]
        
        # 提取基础代码（去掉交易所后缀）
        base_symbol = symbol.split('.')[0]
        url = f"{self.base_url}/hsstock/instrument/{base_symbol}/{self.licence}"
        
        data = self.make_request(url)
        
        if data:
            # API可能返回列表或字典
            if isinstance(data, list) and len(data) > 0:
                self.stock_info_cache[symbol] = data[0]
                return data[0]
            elif isinstance(data, dict):
                self.stock_info_cache[symbol] = data
                return data
        return None
    
    def get_balance_sheet(self, symbol: str, years_back: int = 1) -> List[Dict]:
        """获取资产负债表"""
        # 提取基础代码（去掉交易所后缀）
        if '.' in symbol:
            base_symbol = symbol.split('.')[0]
        else:
            base_symbol = symbol
            
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years_back)
        
        st = start_date.strftime("%Y%m%d")
        et = end_date.strftime("%Y%m%d")
        
        url = f"{self.base_url}/hsstock/financial/balance/{base_symbol}/{self.licence}?st={st}&et={et}"
        data = self.make_request(url)
        
        if data and isinstance(data, list):
            return data
        elif data and isinstance(data, dict):
            # 如果是字典，尝试提取列表
            if 'data' in data and isinstance(data['data'], list):
                return data['data']
        return []
    
    def get_income_statement(self, symbol: str, years_back: int = 5) -> List[Dict]:
        """获取利润表"""
        # 提取基础代码（去掉交易所后缀）
        if '.' in symbol:
            base_symbol = symbol.split('.')[0]
        else:
            base_symbol = symbol
            
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years_back)
        
        st = start_date.strftime("%Y%m%d")
        et = end_date.strftime("%Y%m%d")
        
        url = f"{self.base_url}/hsstock/financial/income/{base_symbol}/{self.licence}?st={st}&et={et}"
        data = self.make_request(url)
        
        if data and isinstance(data, list):
            return data
        elif data and isinstance(data, dict):
            # 如果是字典，尝试提取列表
            if 'data' in data and isinstance(data['data'], list):
                return data['data']
        return []
    
    def get_cash_flow(self, symbol: str, years_back: int = 1) -> List[Dict]:
        """获取现金流量表"""
        # 提取基础代码（去掉交易所后缀）
        if '.' in symbol:
            base_symbol = symbol.split('.')[0]
        else:
            base_symbol = symbol
            
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years_back)
        
        st = start_date.strftime("%Y%m%d")
        et = end_date.strftime("%Y%m%d")
        
        url = f"{self.base_url}/hsstock/financial/cashflow/{base_symbol}/{self.licence}?st={st}&et={et}"
        data = self.make_request(url)
        
        if data and isinstance(data, list):
            return data
        elif data and isinstance(data, dict):
            # 如果是字典，尝试提取列表
            if 'data' in data and isinstance(data['data'], list):
                return data['data']
        return []
    
    def get_financial_indicators(self, symbol: str, years_back: int = 5) -> List[Dict]:
        """获取财务主要指标"""
        # 提取基础代码（去掉交易所后缀）
        if '.' in symbol:
            base_symbol = symbol.split('.')[0]
        else:
            base_symbol = symbol
            
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years_back)
        
        st = start_date.strftime("%Y%m%d")
        et = end_date.strftime("%Y%m%d")
        
        url = f"{self.base_url}/hsstock/financial/pershareindex/{base_symbol}/{self.licence}?st={st}&et={et}"
        data = self.make_request(url)
        
        if data and isinstance(data, list):
            return data
        elif data and isinstance(data, dict):
            # 如果是字典，尝试提取列表
            if 'data' in data and isinstance(data['data'], list):
                return data['data']
        return []
    
    def get_historical_prices(self, symbol: str, years_back: int = 5) -> List[Dict]:
        """获取历史价格数据（日线）"""
        # 提取基础代码（去掉交易所后缀）
        if '.' in symbol:
            base_symbol = symbol.split('.')[0]
        else:
            base_symbol = symbol
            
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years_back)
        
        st = start_date.strftime("%Y%m%d")
        et = end_date.strftime("%Y%m%d")
        
        # 获取日线数据，不复权
        url = f"{self.base_url}/hsstock/history/{base_symbol}/d/n/{self.licence}?st={st}&et={et}"
        data = self.make_request(url)
        
        if data and isinstance(data, list):
            return data
        elif data and isinstance(data, dict):
            # 如果是字典，尝试提取列表
            if 'data' in data and isinstance(data['data'], list):
                return data['data']
        return []
    
    def extract_stock_data(self, symbol: str, name: str, exchange: str) -> Optional[Stock]:
        """提取股票数据"""
        try:
            logger.info(f"处理股票: {name} ({symbol})")
            
            # 创建股票对象
            stock = Stock(symbol=symbol, name=name, exchange=exchange)
            
            # 1. 获取基本信息
            info = self.get_stock_info(symbol)
            if not info:
                logger.warning(f"无法获取 {symbol} 基本信息")
                return None
            
            # 提取价格和其他基本信息
            stock.price = float(info.get('pc', 0) or 0)
            stock.total_shares = float(info.get('tv', 0) or 0)  # 总股本（万股）
            
            # 记录获取到的基本信息
            logger.debug(f"股票 {symbol} 基本信息: 价格={stock.price}, 总股本={stock.total_shares}万股")
            
            # 2. 获取财务指标（主要获取每股收益）
            indicators_data = self.get_financial_indicators(symbol, years_back=2)
            if indicators_data:
                # 获取最新的每股收益
                for indicator in indicators_data[:5]:  # 查看最近几个报告期
                    eps = indicator.get('jbmgsy', 0)
                    if eps and eps != '-':
                        try:
                            stock.eps = float(eps)
                            break
                        except (ValueError, TypeError):
                            continue
                else:
                    # 如果没有找到，使用第一个
                    if indicators_data:
                        eps = indicators_data[0].get('jbmgsy', 0)
                        if eps and eps != '-':
                            try:
                                stock.eps = float(eps)
                            except (ValueError, TypeError):
                                stock.eps = 0
            
            # 3. 获取资产负债表
            balance_data = self.get_balance_sheet(symbol, years_back=1)
            if balance_data:
                latest_balance = balance_data[0]
                stock.current_assets = float(latest_balance.get('ldzchj', 0) or 0)
                stock.total_assets = float(latest_balance.get('zczj', 0) or 0)
                stock.current_debt = float(latest_balance.get('ldfzhj', 0) or 0)
                stock.total_debt = float(latest_balance.get('fzhj', 0) or 0)
                stock.net_assets = float(latest_balance.get('syzqyhj', 0) or 0)
            
            # 4. 获取现金流量表
            cash_flow_data = self.get_cash_flow(symbol, years_back=1)
            if cash_flow_data:
                latest_cash_flow = cash_flow_data[0]
                stock.dividend_cash_paid = float(latest_cash_flow.get('fpglrlhcllxzfdxj', 0) or 0)
            
            # 5. 获取利润表数据（用于历史EPS）
            income_data = self.get_income_statement(symbol, years_back=5)
            if income_data:
                # 提取年度净利润数据
                annual_profits = []
                for item in income_data:
                    jzrq = item.get('jzrq', '')
                    if jzrq and jzrq.endswith('1231'):
                        profit = item.get('jlr', 0)  # 净利润（万元）
                        try:
                            if isinstance(profit, str):
                                profit = float(profit.replace(',', ''))
                            else:
                                profit = float(profit)
                            if profit > 0:
                                annual_profits.append(profit)
                        except (ValueError, TypeError):
                            continue
                
                # 按年份排序并取最近5年
                annual_profits.sort(reverse=True)
                stock.historical_eps = annual_profits[:5]
                
                # 获取利息支出
                for item in income_data[:3]:
                    interest = item.get('cwfy', 0)  # 财务费用作为利息估算
                    try:
                        if isinstance(interest, str):
                            interest = abs(float(interest.replace(',', '')))
                        else:
                            interest = abs(float(interest))
                        if interest > 0:
                            stock.interest_expense = interest
                            break
                    except (ValueError, TypeError):
                        continue
            
            # 6. 获取历史价格数据（用于历史市盈率）
            historical_prices = self.get_historical_prices(symbol, years_back=5)
            if historical_prices:
                # 按交易时间排序
                sorted_prices = sorted(historical_prices, key=lambda x: x.get('t', ''))
                
                # 提取每年12月的价格
                yearly_prices = {}
                for price_data in sorted_prices:
                    trade_time = price_data.get('t', '')
                    if trade_time and '-12-' in trade_time:
                        year = trade_time[:4]
                        close_price = price_data.get('c', 0)
                        try:
                            close_price = float(close_price)
                            if close_price > 0:
                                yearly_prices[year] = close_price
                        except (ValueError, TypeError):
                            continue
                
                stock.historical_prices = yearly_prices
            
            # 7. 计算历史市盈率
            if stock.historical_prices and stock.historical_eps:
                for year, price in stock.historical_prices.items():
                    # 找到对应年份的净利润（转换为每股收益）
                    year_int = int(year)
                    current_year = datetime.now().year
                    
                    # 简单匹配：当前年份-1对应最近的EPS
                    year_diff = current_year - year_int
                    if 0 <= year_diff < len(stock.historical_eps):
                        eps_value = stock.historical_eps[year_diff]
                        # 将净利润转换为每股收益（需要总股本）
                        if stock.total_shares > 0:
                            eps = eps_value / stock.total_shares  # 万元/万股 = 元/股
                            if eps > 0:
                                pe = price / eps
                                stock.historical_pe.append(pe)
            
            # 8. 计算各项指标
            stock.calculate_metrics()
            
            # 记录调试信息
            logger.info(f"{symbol}: 价格={stock.price:.2f}, PE={stock.pe:.2f}, PB={stock.pb:.2f}, "
                       f"股息率={stock.dvr*100:.2f}%, 流动比率={stock.current_ratio:.2f}, "
                       f"总股本={stock.total_shares:.0f}万股")
            
            return stock
            
        except Exception as e:
            logger.error(f"提取股票数据失败 {symbol}: {e}", exc_info=True)
            return None
    
    # ==================== 筛选条件 ====================
    
    def step1_price_limit(self, stock: Stock) -> bool:
        """Step 1: 单股低于200元"""
        return stock.price < 200 and stock.price > 0
    
    def step2_pe_limit(self, stock: Stock) -> bool:
        """Step 2: 市盈率小于20"""
        return stock.pe < 20 and stock.pe > 0
    
    def step3_dividend_yield(self, stock: Stock) -> bool:
        """Step 3: 股息率大于1%"""
        return stock.dvr > 0.01
    
    def step4_price_to_book(self, stock: Stock) -> bool:
        """Step 4: 股价小于资产净值的2/3"""
        return stock.pb < (2/3) and stock.pb > 0
    
    def step5_price_to_net_current_assets(self, stock: Stock) -> bool:
        """Step 5: 股价小于（流动资产-负债）*2/3"""
        if stock.total_shares > 0:
            # 注意：current_assets和total_debt都是万元，total_shares是万股
            # 每股净流动资产 = (流动资产 - 总负债) / 总股本
            net_current_assets_per_share = (stock.current_assets - stock.total_debt) / stock.total_shares
            return (net_current_assets_per_share > 0 and 
                    stock.price < net_current_assets_per_share * (2/3))
        return False
    
    def step6_debt_to_assets(self, stock: Stock) -> bool:
        """Step 6: 总负债小于总资产净值*0.8"""
        if stock.total_assets > 0:
            debt_ratio = stock.total_debt / stock.total_assets
            return debt_ratio < 0.8
        return False
    
    def step7_current_ratio(self, stock: Stock) -> bool:
        """Step 7: 流动比率大于2"""
        return stock.current_ratio > 2
    
    def step8_debt_to_net_current_assets(self, stock: Stock) -> bool:
        """Step 8: 总负债小于净流动资产的2倍"""
        net_current_assets = stock.current_assets - stock.current_debt
        return stock.total_debt < 2 * net_current_assets and net_current_assets > 0
    
    def step9_earnings_growth(self, stock: Stock) -> bool:
        """Step 9: 过去5年的平均年化盈利增长率大于7%"""
        if len(stock.historical_eps) >= 2:
            growth_rates = []
            for i in range(1, len(stock.historical_eps)):
                if stock.historical_eps[i-1] > 0:
                    growth = (stock.historical_eps[i] - stock.historical_eps[i-1]) / abs(stock.historical_eps[i-1])
                    growth_rates.append(growth)
            
            if growth_rates:
                avg_growth = sum(growth_rates) / len(growth_rates)
                return avg_growth > 0.07
        return True  # 数据不足时跳过
    
    def step10_earnings_stability(self, stock: Stock) -> bool:
        """Step 10: 过去5年中不能超过2次的年盈利增长率小于-5%"""
        if len(stock.historical_eps) >= 2:
            negative_growth_count = 0
            for i in range(1, len(stock.historical_eps)):
                if stock.historical_eps[i-1] > 0:
                    growth = (stock.historical_eps[i] - stock.historical_eps[i-1]) / abs(stock.historical_eps[i-1])
                    if growth < -0.05:
                        negative_growth_count += 1
            
            return negative_growth_count <= 2
        return True  # 数据不足时跳过
    
    def step11_historical_pe(self, stock: Stock) -> bool:
        """Step 11: 市盈率小于过去五年最高市盈率的60%"""
        if stock.historical_pe and stock.pe > 0:
            if stock.historical_pe:
                max_historical_pe = max(stock.historical_pe)
                return stock.pe < 0.6 * max_historical_pe
        return True  # 数据不足时跳过
    
    def apply_graham_screen(self, stock: Stock) -> bool:
        """应用格雷厄姆筛选法"""
        # Step 1-10 筛选
        steps_1_10 = [
            self.step1_price_limit,
            self.step2_pe_limit,
            self.step3_dividend_yield,
            self.step4_price_to_book,
            self.step5_price_to_net_current_assets,
            self.step6_debt_to_assets,
            self.step7_current_ratio,
            self.step8_debt_to_net_current_assets,
            self.step9_earnings_growth,
            self.step10_earnings_stability,
        ]
        
        # 记录每一步的结果
        step_results = []
        for step in steps_1_10:
            try:
                result = step(stock)
                step_results.append(result)
            except Exception as e:
                logger.warning(f"{stock.symbol} 筛选步骤出错: {e}")
                step_results.append(False)
        
        # 统计通过的数量
        passed_count = sum(step_results)
        
        # 生成结果字符串
        result_str = ''.join(['T' if r else 'F' for r in step_results])
        stock.step_results = step_results
        stock.step_summary = result_str
        
        # 策略：通过9项或以上才进行Step 11筛选
        if passed_count >= 9:
            logger.info(f"{stock.symbol} 通过前10项中的 {passed_count} 项，进行Step 11筛选")
            try:
                step11_result = self.step11_historical_pe(stock)
                step_results.append(step11_result)
                stock.step_results = step_results
                stock.step_summary = result_str + ('T' if step11_result else 'F')
                return step11_result
            except Exception as e:
                logger.warning(f"{stock.symbol} Step 11筛选出错: {e}")
                return False
        else:
            logger.info(f"{stock.symbol} 仅通过前10项中的 {passed_count} 项，未达到Step 11筛选条件")
            return False
    
    def process_stock(self, stock_info: Dict) -> Optional[Stock]:
        """处理单只股票"""
        symbol = stock_info.get('dm', '')
        name = stock_info.get('mc', '')
        exchange = stock_info.get('jys', 'SZ')
        
        if not symbol:
            return None
        
        stock = self.extract_stock_data(symbol, name, exchange)
        if stock and self.apply_graham_screen(stock):
            logger.info(f"✓ {stock.symbol} {stock.name} 通过筛选")
            return stock
        return None
    
    def run_screening(self) -> List[Stock]:
        """运行筛选"""
        logger.info("开始格雷厄姆筛选...")
        
        # 获取股票列表
        stocks_list = self.get_stock_list()
        if not stocks_list:
            logger.error("无法获取股票列表")
            return []
        
        logger.info(f"共获取 {len(stocks_list)} 支股票，开始处理...")
        
        # 使用线程池并发处理（注意API限制）
        good_stocks = []
        with ThreadPoolExecutor(max_workers=5) as executor:  # 限制并发数
            futures = []
            for stock_info in stocks_list:
                future = executor.submit(self.process_stock, stock_info)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=60)
                    if result:
                        good_stocks.append(result)
                except Exception as e:
                    logger.error(f"处理股票时出错: {e}")
        
        self.good_stocks = good_stocks
        return good_stocks
    
    def display_results(self):
        """显示筛选结果"""
        if not self.good_stocks:
            print("\n" + "="*120)
            print("没有找到符合条件的股票")
            print("="*120)
            return
        
        print("\n" + "="*120)
        print(f"找到 {len(self.good_stocks)} 支符合条件的股票:")
        print("="*120)
        
        # 表头
        header = (
            f"{'代码':<12} {'名称':<20} {'价格':<8} {'PE':<8} {'PB':<8} "
            f"{'股息率':<8} {'流动比率':<10} {'筛选结果':<12} {'通过数':<8}"
        )
        print(header)
        print("-"*120)
        
        # 详细数据
        for stock in self.good_stocks:
            # 计算通过的数量
            passed_count = sum(stock.step_results[:10])  # 前10项
            
            line = (
                f"{stock.symbol:<12} {stock.name:<20} {stock.price:<8.2f} "
                f"{stock.pe:<8.2f} {stock.pb:<8.2f} {stock.dvr*100:<7.2f}% "
                f"{stock.current_ratio:<10.2f} {stock.step_summary:<12} {passed_count:<8}"
            )
            print(line)
        
        print("="*120)
        
        # 显示筛选详情
        print("\n筛选条件详情:")
        print("-"*60)
        conditions = [
            "1. 股价<200元",
            "2. PE<20",
            "3. 股息率>1%",
            "4. PB<2/3",
            "5. 股价<净流动资产2/3",
            "6. 负债率<80%",
            "7. 流动比率>2",
            "8. 总负债<净流动资产2倍",
            "9. 盈利增长率>7%",
            "10. 盈利稳定性",
            "11. PE<历史最高PE60%"
        ]
        
        for i, condition in enumerate(conditions):
            passed_count = 0
            for s in self.good_stocks:
                if i < len(s.step_results) and s.step_results[i]:
                    passed_count += 1
            print(f"{condition:<25} - 通过 {passed_count:>3} 支股票")
        
        # 保存详细结果
        self.save_detailed_results()
    
    def save_detailed_results(self):
        """保存详细结果到CSV和Excel"""
        if not self.good_stocks:
            return
        
        # 准备数据
        data = []
        for stock in self.good_stocks:
            # 计算各项指标
            debt_ratio = stock.total_debt / stock.total_assets if stock.total_assets > 0 else 0
            book_value_per_share = stock.net_assets / stock.total_shares if stock.total_shares > 0 else 0
            net_current_per_share = (stock.current_assets - stock.total_debt) / stock.total_shares if stock.total_shares > 0 else 0
            
            # 计算每股分红
            dividend_per_share = stock.dvr * stock.price if stock.price > 0 else 0
            
            row = {
                '代码': stock.symbol,
                '名称': stock.name,
                '价格(元)': stock.price,
                '市盈率(PE)': stock.pe,
                '市净率(PB)': stock.pb,
                '股息率(%)': stock.dvr * 100,
                '每股分红(元)': dividend_per_share,
                '流动比率': stock.current_ratio,
                '负债率(%)': debt_ratio * 100,
                '每股净资产(元)': book_value_per_share,
                '每股净流动资产(元)': net_current_per_share,
                '总股本(万股)': stock.total_shares,
                '市值(亿元)': stock.market_cap,
                '筛选结果': stock.step_summary,
                '前10项通过数': sum(stock.step_results[:10]),
                'Step1通过': 'T' if stock.step_results[0] else 'F',
                'Step2通过': 'T' if stock.step_results[1] else 'F',
                'Step3通过': 'T' if stock.step_results[2] else 'F',
                'Step4通过': 'T' if stock.step_results[3] else 'F',
                'Step5通过': 'T' if stock.step_results[4] else 'F',
                'Step6通过': 'T' if stock.step_results[5] else 'F',
                'Step7通过': 'T' if stock.step_results[6] else 'F',
                'Step8通过': 'T' if stock.step_results[7] else 'F',
                'Step9通过': 'T' if stock.step_results[8] else 'F',
                'Step10通过': 'T' if stock.step_results[9] else 'F',
                'Step11通过': 'T' if len(stock.step_results) > 10 and stock.step_results[10] else 'F',
            }
            data.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 保存为CSV
        csv_filename = f"graham_screened_stocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        logger.info(f"结果已保存到 {csv_filename}")
        
        # 保存为Excel
        try:
            excel_filename = f"graham_screened_stocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                # 完整结果
                df.to_excel(writer, sheet_name='完整结果', index=False)
                
                # 筛选结果汇总
                summary_data = []
                for i in range(11):
                    step_num = i + 1
                    col_name = f'Step{step_num}通过' if step_num <= 10 else 'Step11通过'
                    passed_count = df[col_name].eq('T').sum() if col_name in df.columns else 0
                    summary_data.append({
                        '筛选步骤': f'Step {step_num}',
                        '通过股票数': passed_count,
                        '通过率(%)': passed_count / len(df) * 100 if len(df) > 0 else 0
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='筛选统计', index=False)
                
                # 优质股票（通过所有步骤）
                perfect_stocks = df[df['前10项通过数'] == 10]
                if not perfect_stocks.empty:
                    perfect_stocks.to_excel(writer, sheet_name='优质股票', index=False)
            
            logger.info(f"详细结果已保存到 {excel_filename}")
        except Exception as e:
            logger.warning(f"无法保存Excel文件: {e}")

def main():
    """主函数"""
    print("="*60)
    print("格雷厄姆价值投资筛选器")
    print("="*60)
    
    # 配置参数
    LICENCE = "licences-78FA-447B-8FCD-E5E0328368CF"  # 请替换为你的有效许可证
    TEST_MODE = False  # 设置为False处理所有股票
    
    print(f"许可证: {LICENCE[:20]}...")
    print(f"测试模式: {TEST_MODE} ({'前50支股票' if TEST_MODE else '所有股票'})")
    print(f"日志文件: logs/graham_screener_*.log")
    print("="*60)
    print("筛选策略: 通过Step1-10中的至少9项，才进行Step11筛选")
    print("="*60)
    
    # 创建筛选器
    screener = GrahamScreener(licence=LICENCE, test_mode=TEST_MODE)
    
    # 运行筛选
    start_time = time.time()
    good_stocks = screener.run_screening()
    end_time = time.time()
    
    # 显示结果
    screener.display_results()
    
    logger.info(f"筛选完成，耗时: {end_time - start_time:.2f} 秒")
    
    print(f"\n筛选完成，耗时: {(end_time - start_time)/60:.2f} 分钟")
    print(f"日志已保存到: {log_filename}")
    
    # 提示保存结果
    if screener.good_stocks:
        print(f"\n详细结果已保存到CSV和Excel文件")
        print("建议使用Excel打开文件查看完整筛选结果")

if __name__ == "__main__":
    main()
