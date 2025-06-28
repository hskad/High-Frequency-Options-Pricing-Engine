
# Python Wrapper for High-Frequency Options Pricing Engine
# This module provides a Python interface to the C++ pricing engine
# and integrates with QuantLib for additional functionality

import numpy as np
import pandas as pd
import QuantLib as ql
from typing import Dict, List, Optional, Union, Tuple
import time
import json
from dataclasses import dataclass
from enum import Enum
import warnings

class OptionType(Enum):
    CALL = "call"
    PUT = "put"

class OptionStyle(Enum):
    EUROPEAN = "european"
    AMERICAN = "american"

class PricingMethod(Enum):
    MONTE_CARLO = "monte_carlo"
    FINITE_DIFFERENCE = "finite_difference"
    BLACK_SCHOLES = "black_scholes"
    QUANTLIB = "quantlib"

@dataclass
class MarketData:
    """Market data structure for option pricing"""
    underlying_price: float
    risk_free_rate: float
    dividend_yield: float = 0.0
    volatility: float = 0.2
    time_to_expiration: float = 1.0

    def to_dict(self) -> Dict:
        return {
            'underlying_price': self.underlying_price,
            'risk_free_rate': self.risk_free_rate,
            'dividend_yield': self.dividend_yield,
            'volatility': self.volatility,
            'time_to_expiration': self.time_to_expiration
        }

@dataclass
class OptionContract:
    """Option contract specification"""
    underlying: str
    strike_price: float
    option_type: OptionType
    option_style: OptionStyle
    time_to_expiration: float

    def to_dict(self) -> Dict:
        return {
            'underlying': self.underlying,
            'strike_price': self.strike_price,
            'option_type': self.option_type.value,
            'option_style': self.option_style.value,
            'time_to_expiration': self.time_to_expiration
        }

@dataclass
class Greeks:
    """Option Greeks"""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'rho': self.rho
        }

@dataclass
class PricingResult:
    """Option pricing result"""
    option_price: float
    greeks: Greeks
    implied_volatility: float = 0.0
    calculation_time: float = 0.0  # in milliseconds
    convergence_steps: int = 0
    method: str = ""
    confidence_interval: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict:
        result = {
            'option_price': self.option_price,
            'greeks': self.greeks.to_dict(),
            'implied_volatility': self.implied_volatility,
            'calculation_time': self.calculation_time,
            'convergence_steps': self.convergence_steps,
            'method': self.method
        }
        if self.confidence_interval:
            result['confidence_interval'] = self.confidence_interval
        return result

class QuantLibEngine:
    """QuantLib-based pricing engine"""

    def __init__(self):
        self.calendar = ql.UnitedStates()
        self.day_count = ql.Actual365Fixed()

    def price(self, option: OptionContract, market: MarketData) -> PricingResult:
        """Price option using QuantLib"""
        start_time = time.time()

        try:
            # Set up QuantLib date handling
            calculation_date = ql.Date.todaysDate()
            ql.Settings.instance().evaluationDate = calculation_date

            # Convert time to expiration to maturity date
            expiry_date = calculation_date + int(option.time_to_expiration * 365)

            # Create option payoff
            if option.option_type == OptionType.CALL:
                payoff = ql.PlainVanillaPayoff(ql.Option.Call, option.strike_price)
            else:
                payoff = ql.PlainVanillaPayoff(ql.Option.Put, option.strike_price)

            # Create exercise
            if option.option_style == OptionStyle.EUROPEAN:
                exercise = ql.EuropeanExercise(expiry_date)
            else:
                exercise = ql.AmericanExercise(calculation_date, expiry_date)

            # Create option
            ql_option = ql.VanillaOption(payoff, exercise)

            # Market data
            underlying_handle = ql.QuoteHandle(ql.SimpleQuote(market.underlying_price))
            flat_ts = ql.YieldTermStructureHandle(
                ql.FlatForward(calculation_date, market.risk_free_rate, self.day_count)
            )
            dividend_yield = ql.YieldTermStructureHandle(
                ql.FlatForward(calculation_date, market.dividend_yield, self.day_count)
            )
            flat_vol_ts = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(calculation_date, self.calendar, 
                                   market.volatility, self.day_count)
            )

            # Create Black-Scholes-Merton process
            bsm_process = ql.BlackScholesMertonProcess(
                underlying_handle, dividend_yield, flat_ts, flat_vol_ts
            )

            # Choose pricing engine
            if option.option_style == OptionStyle.EUROPEAN:
                pricing_engine = ql.AnalyticEuropeanEngine(bsm_process)
            else:
                pricing_engine = ql.BinomialVanillaEngine(bsm_process, "crr", 100)

            ql_option.setPricingEngine(pricing_engine)

            # Calculate price and Greeks
            option_price = ql_option.NPV()

            greeks = Greeks(
                delta=ql_option.delta(),
                gamma=ql_option.gamma(),
                theta=ql_option.theta(),
                vega=ql_option.vega(),
                rho=ql_option.rho()
            )

            calc_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            return PricingResult(
                option_price=option_price,
                greeks=greeks,
                calculation_time=calc_time,
                method="QuantLib",
                convergence_steps=1
            )

        except Exception as e:
            warnings.warn(f"QuantLib pricing failed: {str(e)}")
            return PricingResult(
                option_price=0.0,
                greeks=Greeks(),
                calculation_time=(time.time() - start_time) * 1000,
                method="QuantLib (Failed)"
            )

class MonteCarloEngine:
    """Monte Carlo pricing engine"""

    def __init__(self, num_paths: int = 100000, time_steps: int = 252, seed: int = 12345):
        self.num_paths = num_paths
        self.time_steps = time_steps
        self.seed = seed
        np.random.seed(seed)

    def price(self, option: OptionContract, market: MarketData) -> PricingResult:
        """Price option using Monte Carlo simulation"""
        start_time = time.time()

        # Time parameters
        dt = market.time_to_expiration / self.time_steps
        drift = (market.risk_free_rate - market.dividend_yield - 
                0.5 * market.volatility**2) * dt
        diffusion = market.volatility * np.sqrt(dt)

        # Generate random paths
        random_shocks = np.random.normal(0, 1, (self.num_paths, self.time_steps))

        # Initialize price paths
        prices = np.zeros((self.num_paths, self.time_steps + 1))
        prices[:, 0] = market.underlying_price

        # Generate price paths using geometric Brownian motion
        for t in range(1, self.time_steps + 1):
            prices[:, t] = prices[:, t-1] * np.exp(drift + diffusion * random_shocks[:, t-1])

        # Calculate payoffs
        final_prices = prices[:, -1]
        if option.option_type == OptionType.CALL:
            payoffs = np.maximum(final_prices - option.strike_price, 0)
        else:
            payoffs = np.maximum(option.strike_price - final_prices, 0)

        # Discount to present value
        discount_factor = np.exp(-market.risk_free_rate * market.time_to_expiration)
        option_price = np.mean(payoffs) * discount_factor

        # Calculate confidence interval
        std_error = np.std(payoffs) / np.sqrt(self.num_paths) * discount_factor
        confidence_interval = (
            option_price - 1.96 * std_error,
            option_price + 1.96 * std_error
        )

        # Estimate Greeks using finite differences (simplified)
        greeks = self._estimate_greeks(option, market, option_price)

        calc_time = (time.time() - start_time) * 1000

        return PricingResult(
            option_price=option_price,
            greeks=greeks,
            calculation_time=calc_time,
            convergence_steps=self.num_paths,
            method="Monte Carlo",
            confidence_interval=confidence_interval
        )

    def _estimate_greeks(self, option: OptionContract, market: MarketData, 
                        base_price: float) -> Greeks:
        """Estimate Greeks using finite differences"""
        bump = 0.01

        # Delta
        market_up = MarketData(
            underlying_price=market.underlying_price + bump,
            risk_free_rate=market.risk_free_rate,
            dividend_yield=market.dividend_yield,
            volatility=market.volatility,
            time_to_expiration=market.time_to_expiration
        )
        price_up = self._quick_price(option, market_up)
        delta = (price_up - base_price) / bump

        # Approximate other Greeks (simplified)
        return Greeks(delta=delta, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)

    def _quick_price(self, option: OptionContract, market: MarketData) -> float:
        """Quick pricing for Greeks calculation"""
        # Use fewer paths for Greeks estimation
        num_paths = min(self.num_paths // 10, 10000)

        dt = market.time_to_expiration / self.time_steps
        drift = (market.risk_free_rate - market.dividend_yield - 
                0.5 * market.volatility**2) * dt
        diffusion = market.volatility * np.sqrt(dt)

        random_shocks = np.random.normal(0, 1, (num_paths, self.time_steps))

        prices = np.zeros((num_paths, self.time_steps + 1))
        prices[:, 0] = market.underlying_price

        for t in range(1, self.time_steps + 1):
            prices[:, t] = prices[:, t-1] * np.exp(drift + diffusion * random_shocks[:, t-1])

        final_prices = prices[:, -1]
        if option.option_type == OptionType.CALL:
            payoffs = np.maximum(final_prices - option.strike_price, 0)
        else:
            payoffs = np.maximum(option.strike_price - final_prices, 0)

        discount_factor = np.exp(-market.risk_free_rate * market.time_to_expiration)
        return np.mean(payoffs) * discount_factor

class BlackScholesEngine:
    """Analytical Black-Scholes pricing engine"""

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Cumulative distribution function for standard normal distribution"""
        from scipy.stats import norm
        return norm.cdf(x)

    @staticmethod
    def _normal_pdf(x: float) -> float:
        """Probability density function for standard normal distribution"""
        return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x * x)

    def price(self, option: OptionContract, market: MarketData) -> PricingResult:
        """Price option using Black-Scholes formula"""
        start_time = time.time()

        if option.option_style == OptionStyle.AMERICAN:
            warnings.warn("Black-Scholes does not support American options")
            return PricingResult(
                option_price=0.0,
                greeks=Greeks(),
                calculation_time=(time.time() - start_time) * 1000,
                method="Black-Scholes (Unsupported)"
            )

        S = market.underlying_price
        K = option.strike_price
        r = market.risk_free_rate
        q = market.dividend_yield
        sigma = market.volatility
        T = market.time_to_expiration

        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Calculate option price
        if option.option_type == OptionType.CALL:
            option_price = (S * np.exp(-q * T) * self._normal_cdf(d1) - 
                           K * np.exp(-r * T) * self._normal_cdf(d2))
        else:
            option_price = (K * np.exp(-r * T) * self._normal_cdf(-d2) - 
                           S * np.exp(-q * T) * self._normal_cdf(-d1))

        # Calculate Greeks
        greeks = self._calculate_greeks(S, K, r, q, sigma, T, d1, d2, option.option_type)

        calc_time = (time.time() - start_time) * 1000

        return PricingResult(
            option_price=option_price,
            greeks=greeks,
            calculation_time=calc_time,
            convergence_steps=1,
            method="Black-Scholes Analytical"
        )

    def _calculate_greeks(self, S: float, K: float, r: float, q: float, 
                         sigma: float, T: float, d1: float, d2: float, 
                         option_type: OptionType) -> Greeks:
        """Calculate option Greeks"""
        Nd1 = self._normal_cdf(d1)
        nd1 = self._normal_pdf(d1)
        Nd2 = self._normal_cdf(d2)

        # Delta
        if option_type == OptionType.CALL:
            delta = np.exp(-q * T) * Nd1
        else:
            delta = np.exp(-q * T) * (Nd1 - 1.0)

        # Gamma
        gamma = (np.exp(-q * T) * nd1) / (S * sigma * np.sqrt(T))

        # Theta
        theta_part1 = -(S * np.exp(-q * T) * nd1 * sigma) / (2.0 * np.sqrt(T))
        theta_part2 = r * K * np.exp(-r * T)
        theta_part3 = q * S * np.exp(-q * T)

        if option_type == OptionType.CALL:
            theta = (theta_part1 - theta_part2 * Nd2 + theta_part3 * Nd1) / 365.0
        else:
            theta = (theta_part1 + theta_part2 * (1.0 - Nd2) - 
                    theta_part3 * (1.0 - Nd1)) / 365.0

        # Vega
        vega = S * np.exp(-q * T) * nd1 * np.sqrt(T) / 100.0

        # Rho
        if option_type == OptionType.CALL:
            rho = K * T * np.exp(-r * T) * Nd2 / 100.0
        else:
            rho = -K * T * np.exp(-r * T) * (1.0 - Nd2) / 100.0

        return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)

class HFPricingManager:
    """High-frequency pricing manager"""

    def __init__(self):
        self.engines = {
            PricingMethod.MONTE_CARLO: MonteCarloEngine(),
            PricingMethod.BLACK_SCHOLES: BlackScholesEngine(),
            PricingMethod.QUANTLIB: QuantLibEngine()
        }
        self.result_cache = {}
        self.caching_enabled = False

    def price_option(self, option: OptionContract, market: MarketData, 
                    method: PricingMethod) -> PricingResult:
        """Price a single option"""
        if method not in self.engines:
            raise ValueError(f"Unsupported pricing method: {method}")

        engine = self.engines[method]
        return engine.price(option, market)

    def price_option_chain(self, options: List[OptionContract], 
                          market: MarketData, method: PricingMethod) -> List[PricingResult]:
        """Price multiple options"""
        results = []
        for option in options:
            result = self.price_option(option, market, method)
            results.append(result)
        return results

    def compare_methods(self, option: OptionContract, 
                       market: MarketData) -> Dict[str, PricingResult]:
        """Compare results from different pricing methods"""
        results = {}
        for method in [PricingMethod.BLACK_SCHOLES, PricingMethod.MONTE_CARLO, 
                      PricingMethod.QUANTLIB]:
            try:
                results[method.value] = self.price_option(option, market, method)
            except Exception as e:
                print(f"Failed to price with {method.value}: {str(e)}")
        return results

    def calibrate_implied_volatility(self, option: OptionContract, 
                                   market: MarketData, market_price: float,
                                   tolerance: float = 1e-6) -> float:
        """Calibrate implied volatility using Newton-Raphson method"""
        vol = 0.2  # Initial guess

        for _ in range(100):  # Maximum iterations
            temp_market = MarketData(
                underlying_price=market.underlying_price,
                risk_free_rate=market.risk_free_rate,
                dividend_yield=market.dividend_yield,
                volatility=vol,
                time_to_expiration=market.time_to_expiration
            )

            result = self.price_option(option, temp_market, PricingMethod.BLACK_SCHOLES)
            price_diff = result.option_price - market_price

            if abs(price_diff) < tolerance:
                return vol

            vega = result.greeks.vega * 100.0  # Convert from percentage

            if abs(vega) < 1e-10:
                break

            vol = vol - price_diff / vega
            vol = max(0.001, min(vol, 5.0))  # Keep vol positive and reasonable

        return vol

    def generate_volatility_surface(self, strikes: List[float], 
                                   expiries: List[float], 
                                   market: MarketData) -> pd.DataFrame:
        """Generate implied volatility surface"""
        surface_data = []

        for expiry in expiries:
            for strike in strikes:
                option = OptionContract(
                    underlying="UNDERLYING",
                    strike_price=strike,
                    option_type=OptionType.CALL,
                    option_style=OptionStyle.EUROPEAN,
                    time_to_expiration=expiry
                )

                # Calculate theoretical price
                result = self.price_option(option, market, PricingMethod.BLACK_SCHOLES)

                surface_data.append({
                    'strike': strike,
                    'expiry': expiry,
                    'moneyness': strike / market.underlying_price,
                    'implied_vol': market.volatility,
                    'option_price': result.option_price,
                    'delta': result.greeks.delta
                })

        return pd.DataFrame(surface_data)

# Example usage and testing functions
def example_usage():
    """Example usage of the pricing engine"""

    # Define market data
    market = MarketData(
        underlying_price=100.0,
        risk_free_rate=0.05,
        dividend_yield=0.02,
        volatility=0.2,
        time_to_expiration=1.0
    )

    # Define option contract
    option = OptionContract(
        underlying="AAPL",
        strike_price=100.0,
        option_type=OptionType.CALL,
        option_style=OptionStyle.EUROPEAN,
        time_to_expiration=1.0
    )

    # Create pricing manager
    manager = HFPricingManager()

    # Compare different methods
    results = manager.compare_methods(option, market)

    print("Option Pricing Results:")
    print("=" * 50)
    for method, result in results.items():
        print(f"{method:20s}: ${result.option_price:.4f} (Time: {result.calculation_time:.2f}ms)")
        print(f"{'':20s}  Delta: {result.greeks.delta:.4f}")
        print()

    return results

if __name__ == "__main__":
    example_usage()
