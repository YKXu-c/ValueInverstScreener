import requests
import time
import json
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Stock:
    """股票数据类"""
    symbol: str
    name: str
    price: float = 0.0
    pe: float = 0.0
    pb: float = 0.0
    dvr: float = 0.0  # 股息率
    total_debt: float = 0.0  # 总负债
    current_debt: float = 0.0  # 流动负债
    current_assets: float = 0.0  # 流动资产
    current_ratio: float = 0.0  # 流动比率
    net_assets: float = 0.0  # 净资产
    total_assets: float = 0.0  # 总资产
    eps: float = 0.0  # 每股收益
    total_shares: float = 0.0  # 总股本（万股转换为股）
    dividend_cash_paid: float = 0.0  # 分配股利、利润或偿付利息支付的现金（万元）
    interest_expense: float = 0.0  # 利息支出（从利润表获取）
    historical_pe: List[float] = None  # 历史市盈率
    historical_eps: List[float] = None  # 历史每股收益
    
    def __post_init__(self):
        if self.historical_pe is None:
            self.historical_pe = []
        if self.historical_eps is None:
            self.historical_eps = []

class GrahamScreener:
    """格雷厄姆筛选器"""
    
    def __init__(self, licence: str, test_mode: bool = True):
        self.licence = licence
        self.test_mode = test_mode
        self.base_url = "https://api.mairuiapi.com"
        self.request_counter = 0
        self.request_limit = 990
        self.good_stocks = []
        self.all_stocks_data = []
        
    def make_request(self, url: str) -> Optional[Dict]:
        """发送API请求并处理限制"""
        if self.request_counter >= self.request_limit:
            logger.info(f"达到请求限制 {self.request_limit}，等待60秒...")
            time.sleep(60)
            self.request_counter = 0
            
        try:
            response = requests.get(url, timeout=30)
            self.request_counter += 1
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"请求失败 {url}: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"请求异常 {url}: {e}")
            return None
    
    def get_stock_list(self) -> List[Dict]:
        """获取股票列表"""
        url = f"{self.base_url}/hslt/list/{self.licence}"
        logger.info("获取股票列表...")
        
        data = self.make_request(url)
        if data and isinstance(data, list):
            if self.test_mode:
                return data[:50]  # 测试模式只取前50支
            return data
        return []
    
    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        """获取股票基本信息"""
        url = f"{self.base_url}/hsstock/instrument/{symbol}/{self.licence}"
        return self.make_request(url)
    
    def get_balance_sheet(self, symbol: str, years_back: int = 1) -> Optional[Dict]:
        """获取资产负债表"""
        # 获取最近几年的数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*years_back)
        
        st = start_date.strftime("%Y%m%d")
        et = end_date.strftime("%Y%m%d")
        
        url = f"{self.base_url}/hsstock/financial/balance/{symbol}/{self.licence}?st={st}&et={et}"
        return self.make_request(url)
    
    def get_income_statement(self, symbol: str) -> Optional[Dict]:
        """获取利润表（最近5年）"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*5)
        
        st = start_date.strftime("%Y%m%d")
        et = end_date.strftime("%Y%m%d")
        
        url = f"{self.base_url}/hsstock/financial/income/{symbol}/{self.licence}?st={st}&et={et}"
        return self.make_request(url)
    
    def get_financial_indicators(self, symbol: str) -> Optional[Dict]:
        """获取财务指标"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*5)
        
        st = start_date.strftime("%Y%m%d")
        et = end_date.strftime("%Y%m%d")
        
        url = f"{self.base_url}/hsstock/financial/pershareindex/{symbol}/{self.licence}?st={st}&et={et}"
        return self.make_request(url)
    
    def get_cash_flow(self, symbol: str) -> Optional[Dict]:
        """获取现金流量表（最近1年）"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        st = start_date.strftime("%Y%m%d")
        et = end_date.strftime("%Y%m%d")
        
        url = f"{self.base_url}/hsstock/financial/cashflow/{symbol}/{self.licence}?st={st}&et={et}"
        return self.make_request(url)
    
    def extract_stock_data(self, symbol: str, name: str, exchange: str) -> Optional[Stock]:
        """提取股票数据"""
        try:
            stock = Stock(symbol=f"{symbol}.{exchange.upper()}", name=name)
            
            # 1. 获取基本信息
            info = self.get_stock_info(stock.symbol)
            if not info or not isinstance(info, list) or len(info) == 0:
                return None
            
            info_data = info[0]
            stock.price = info_data.get('pc', 0)
            
            # 获取总股本（单位：万股 -> 转换为股）
            total_shares_w = info_data.get('tv', 0)  # tv:总股本（万股）
            stock.total_shares = total_shares_w * 10000  # 转换为股
            
            # 2. 获取资产负债表
            balance = self.get_balance_sheet(stock.symbol)
            if balance and isinstance(balance, list) and len(balance) > 0:
                # 取最新的资产负债表数据
                latest_balance = balance[0]
                
                # 提取关键字段
                stock.current_assets = latest_balance.get('ldzchj', 0)  # 流动资产合计（万元）
                stock.total_assets = latest_balance.get('zczj', 0)      # 资产总计（万元）
                stock.current_debt = latest_balance.get('ldfzhj', 0)   # 流动负债合计（万元）
                stock.total_debt = latest_balance.get('fzhj', 0)       # 负债合计（万元）
                stock.net_assets = latest_balance.get('syzqyhj', 0)    # 所有者权益合计（万元）
                
                # 计算流动比率
                if stock.current_debt > 0:
                    stock.current_ratio = stock.current_assets / stock.current_debt
            
            # 3. 获取现金流量表
            cash_flow = self.get_cash_flow(stock.symbol)
            if cash_flow and isinstance(cash_flow, list) and len(cash_flow) > 0:
                # 取最新的现金流量表数据
                latest_cash_flow = cash_flow[0]
                
                # 提取分配股利、利润或偿付利息支付的现金（万元）
                stock.dividend_cash_paid = latest_cash_flow.get('fpglrlhcllxzfdxj', 0)
                
                logger.debug(f"{stock.symbol}: 分配股利、利润或偿付利息支付的现金 = {stock.dividend_cash_paid:.2f}万元")
            
            # 4. 获取财务指标
            indicators = self.get_financial_indicators(stock.symbol)
            if indicators and isinstance(indicators, list):
                # 取最新的财务指标
                for indicator in indicators:
                    jzrq = indicator.get('jzrq', '')
                    # 只取年报数据（通常12-31）
                    if jzrq and jzrq.endswith('1231'):
                        stock.eps = indicator.get('jbmgsy', 0)  # 基本每股收益（元）
                        
                        # 计算市盈率 (避免除以0)
                        if stock.eps > 0:
                            stock.pe = stock.price / stock.eps
                        
                        # 计算市净率
                        # 需要计算每股净资产
                        if stock.total_shares > 0 and stock.net_assets > 0:
                            # 注意单位转换：net_assets是万元，total_shares是股
                            book_value_per_share = (stock.net_assets * 10000) / stock.total_shares  # 转换为元/股
                            if book_value_per_share > 0:
                                stock.pb = stock.price / book_value_per_share
                        break
            
            # 5. 获取利润表数据用于盈利增长率计算，同时获取利息支出
            income = self.get_income_statement(stock.symbol)
            if income and isinstance(income, list):
                # 提取过去5年的净利润数据和最近一年的利息支出
                yearly_profits = []
                for item in income:
                    jzrq = item.get('jzrq', '')
                    
                    # 取最近一年的利息支出（cwfy:财务费用，包含利息支出）
                    # 注意：财务费用不完全是利息支出，但可作为近似
                    if jzrq and not yearly_profits:  # 只取最新的
                        stock.interest_expense = abs(item.get('cwfy', 0))  # 取绝对值
                    
                    if jzrq and jzrq.endswith('1231'):  # 年报数据
                        profit = item.get('jlr', 0)  # 净利润（万元）
                        if profit > 0:
                            yearly_profits.append((jzrq[:4], profit))
                
                # 按年份排序并取最近5年
                yearly_profits.sort(key=lambda x: x[0], reverse=True)
                stock.historical_eps = [profit for _, profit in yearly_profits[:5]]
                
                logger.debug(f"{stock.symbol}: 利息支出 ≈ {stock.interest_expense:.2f}万元")
            
            # 6. 计算股息率
            self.calculate_dividend_yield(stock)
            
            return stock
            
        except Exception as e:
            logger.error(f"提取股票数据失败 {symbol}: {e}")
            return None
    
    def calculate_dividend_yield(self, stock: Stock):
        """计算股息率"""
        # 股息率 = (分配股利现金 - 估算的利息支出) / 总股本 / 股价
        
        if stock.price <= 0 or stock.total_shares <= 0:
            stock.dvr = 0
            return
        
        if stock.dividend_cash_paid <= 0:
            stock.dvr = 0
            return
        
        # 方法1：从分配股利、利润或偿付利息支付的现金中分离出股利部分
        # 保守估计，假设其中的利息支出部分不超过总支付现金的40%
        
        # 估算利息支出部分
        if stock.interest_expense > 0:
            # 使用利润表中的财务费用作为利息支出参考
            estimated_interest_payment = min(stock.interest_expense, stock.dividend_cash_paid * 0.5)
        else:
            # 没有利息数据，保守假设30%用于支付利息
            estimated_interest_payment = stock.dividend_cash_paid * 0.3
        
        # 估算用于股利的现金
        estimated_dividend_cash = max(0, stock.dividend_cash_paid - estimated_interest_payment)
        
        # 转换为每股分红（元）
        # 注意单位：estimated_dividend_cash是万元，total_shares是股
        dividend_per_share = (estimated_dividend_cash * 10000) / stock.total_shares  # 元/股
        
        # 计算股息率
        stock.dvr = dividend_per_share / stock.price if stock.price > 0 else 0
        
        logger.debug(f"{stock.symbol}: 估算股利现金={estimated_dividend_cash:.2f}万元, "
                    f"每股分红={dividend_per_share:.4f}元, 股息率={stock.dvr*100:.2f}%")
    
    def step1_price_limit(self, stock: Stock) -> bool:
        """Step 1: 单股低于200元"""
        return stock.price < 200 and stock.price > 0
    
    def step2_pe_limit(self, stock: Stock) -> bool:
        """Step 2: 市盈率小于20 (使用jbmgsy/price > 5%代替)"""
        # jbmgsy/price > 5% 等价于 price/jbmgsy < 20
        # 即市盈率 < 20
        return stock.pe < 20 and stock.pe > 0
    
    def step3_dividend_yield(self, stock: Stock) -> bool:
        """Step 3: 股息率大于1%"""
        # 股息率大于1%的条件: dvr > 0.01
        result = stock.dvr > 0.01
        
        if not result and stock.dvr > 0:
            logger.debug(f"{stock.symbol}: 股息率{stock.dvr*100:.2f}%未达到1%要求")
        
        return result
    
    def step4_price_to_book(self, stock: Stock) -> bool:
        """Step 4: 股价小于资产净值的2/3"""
        # 股价 < 净资产/总股本 * 2/3
        # 等价于 pb < 2/3
        return stock.pb < (2/3) and stock.pb > 0
    
    def step5_price_to_net_current_assets(self, stock: Stock) -> bool:
        """Step 5: 股价小于（流动资产-负债）*2/3"""
        # 原条件: 股价 < (流动资产 - 总负债) * 2/3 / 总股本
        # 即: 股价 < (流动资产 - 总负债) / 总股本 * 2/3
        
        # 注意单位转换：current_assets和total_debt是万元，total_shares是股
        if (stock.total_shares > 0 and 
            stock.current_assets > 0):
            
            # 计算每股净流动资产（转换为元/股）
            # (current_assets - total_debt)是万元，乘以10000转换为元
            net_current_assets_per_share = ((stock.current_assets - stock.total_debt) * 10000) / stock.total_shares
            
            # 检查条件: 股价 < 每股净流动资产 * 2/3
            # 并且每股净流动资产必须为正数才有意义
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
        if stock.current_assets > 0 and stock.current_debt > 0:
            net_current_assets = stock.current_assets - stock.current_debt
            return stock.total_debt < 2 * net_current_assets
        return False
    
    def step9_earnings_growth(self, stock: Stock) -> bool:
        """Step 9: 过去5年的平均年化盈利增长率大于7%"""
        if len(stock.historical_eps) >= 2:
            # 计算年化增长率
            eps_growth_rates = []
            for i in range(1, len(stock.historical_eps)):
                if stock.historical_eps[i-1] > 0:
                    growth = (stock.historical_eps[i] - stock.historical_eps[i-1]) / stock.historical_eps[i-1]
                    eps_growth_rates.append(growth)
            
            if eps_growth_rates:
                avg_growth = sum(eps_growth_rates) / len(eps_growth_rates)
                return avg_growth > 0.07
        return False
    
    def step10_earnings_stability(self, stock: Stock) -> bool:
        """Step 10: 过去5年中不能超过2次的年盈利增长率小于-5%"""
        if len(stock.historical_eps) >= 2:
            negative_growth_count = 0
            for i in range(1, len(stock.historical_eps)):
                if stock.historical_eps[i-1] > 0:
                    growth = (stock.historical_eps[i] - stock.historical_eps[i-1]) / stock.historical_eps[i-1]
                    if growth < -0.05:
                        negative_growth_count += 1
            
            return negative_growth_count <= 2
        return False
    
    def step11_historical_pe(self, stock: Stock) -> bool:
        """Step 11: 市盈率小于过去五年最高市盈率的60%"""
        if stock.historical_pe and len(stock.historical_pe) > 0:
            max_historical_pe = max(stock.historical_pe)
            return stock.pe < 0.6 * max_historical_pe
        return True  # 如果没有历史数据，默认通过
    
    def apply_graham_screen(self, stock: Stock) -> bool:
        """应用格雷厄姆筛选法"""
        steps = [
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
            self.step11_historical_pe,
        ]
        
        for i, step in enumerate(steps, 1):
            if not step(stock):
                logger.debug(f"{stock.symbol} 未通过第 {i} 步筛选")
                return False
        
        return True
    
    def process_stock(self, stock_info: Dict) -> Optional[Stock]:
        """处理单只股票"""
        symbol = stock_info.get('dm', '')
        name = stock_info.get('mc', '')
        exchange = stock_info.get('jys', 'sz')
        
        if not symbol:
            return None
        
        logger.info(f"处理股票: {name} ({symbol})")
        
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
        
        logger.info(f"共获取 {len(stocks_list)} 支股票")
        
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
            print("没有找到符合条件的股票")
            return
        
        print(f"\n找到 {len(self.good_stocks)} 支符合条件的股票:")
        print("-" * 100)
        print(f"{'代码':<10} {'名称':<20} {'价格':<8} {'市盈率':<8} {'市净率':<8} {'流动比率':<10} {'股息率':<10}")
        print("-" * 100)
        
        for stock in self.good_stocks:
            print(f"{stock.symbol:<10} {stock.name:<20} {stock.price:<8.2f} "
                  f"{stock.pe:<8.2f} {stock.pb:<8.2f} {stock.current_ratio:<10.2f} "
                  f"{stock.dvr*100:<9.2f}%")
        
        # 保存结果到CSV
        self.save_results_to_csv()
    
    def save_results_to_csv(self, filename: str = "graham_screened_stocks.csv"):
        """保存结果到CSV文件"""
        if not self.good_stocks:
            return
        
        data = []
        for stock in self.good_stocks:
            data.append({
                '代码': stock.symbol,
                '名称': stock.name,
                '价格': stock.price,
                '市盈率': stock.pe,
                '市净率': stock.pb,
                '流动比率': stock.current_ratio,
                '股息率': stock.dvr * 100,  # 转换为百分比
                '总负债/总资产': stock.total_debt/stock.total_assets if stock.total_assets > 0 else 0,
                '每股收益': stock.eps,
                '每股净资产': (stock.net_assets * 10000) / stock.total_shares if stock.total_shares > 0 else 0,
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info(f"结果已保存到 {filename}")

def main():
    """主函数"""
    # 替换为你的licence
    LICENCE = "babababalalalala"
    
    # 创建筛选器（test_mode=True 只处理前50支股票与否）
    screener = GrahamScreener(licence=LICENCE, test_mode=False)
    
    # 运行筛选
    start_time = time.time()
    good_stocks = screener.run_screening()
    end_time = time.time()
    
    # 显示结果
    screener.display_results()
    
    logger.info(f"筛选完成，耗时: {end_time - start_time:.2f} 秒")
    logger.info(f"总请求次数: {screener.request_counter}")

if __name__ == "__main__":
    main()
