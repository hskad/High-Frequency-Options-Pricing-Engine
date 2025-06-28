
# Flask API Backend for High-Frequency Options Pricing Engine
# This provides REST API endpoints for the options pricing functionality

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
import os

# Import our pricing engine (assuming it's in the same directory)
try:
    from python_pricing_engine import (
        HFPricingManager, OptionContract, MarketData, OptionType, 
        OptionStyle, PricingMethod, PricingResult
    )
except ImportError:
    print("Warning: Could not import pricing engine. Mock implementation will be used.")

    # Mock classes for demonstration
    class OptionType:
        CALL = "call"
        PUT = "put"

    class OptionStyle:
        EUROPEAN = "european"
        AMERICAN = "american"

    class PricingMethod:
        BLACK_SCHOLES = "black_scholes"
        MONTE_CARLO = "monte_carlo"
        QUANTLIB = "quantlib"

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pricing manager instance
pricing_manager = None

try:
    pricing_manager = HFPricingManager()
    logger.info("Pricing manager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize pricing manager: {e}")

# Mock data for demonstration
SAMPLE_MARKET_DATA = {
    "underlying_price": 100.0,
    "risk_free_rate": 0.05,
    "dividend_yield": 0.02,
    "volatility": 0.2,
    "time_to_expiration": 1.0
}

SAMPLE_OPTIONS = [
    {
        "underlying": "AAPL",
        "strike_price": 95.0,
        "option_type": "call",
        "option_style": "european",
        "time_to_expiration": 0.25
    },
    {
        "underlying": "AAPL", 
        "strike_price": 100.0,
        "option_type": "call",
        "option_style": "european",
        "time_to_expiration": 0.5
    },
    {
        "underlying": "AAPL",
        "strike_price": 105.0,
        "option_type": "put",
        "option_style": "european", 
        "time_to_expiration": 1.0
    }
]

def create_mock_result(price: float, method: str) -> Dict:
    """Create a mock pricing result for demonstration"""
    return {
        "option_price": price,
        "greeks": {
            "delta": np.random.uniform(0.3, 0.7),
            "gamma": np.random.uniform(0.01, 0.05),
            "theta": np.random.uniform(-0.05, -0.01),
            "vega": np.random.uniform(0.1, 0.3),
            "rho": np.random.uniform(0.05, 0.15)
        },
        "implied_volatility": 0.2,
        "calculation_time": np.random.uniform(0.5, 50.0),
        "convergence_steps": np.random.randint(1, 100000),
        "method": method
    }

@app.route('/')
def home():
    """Home page with API documentation"""
    return jsonify({
        "message": "High-Frequency Options Pricing Engine API",
        "version": "1.0",
        "endpoints": {
            "/api/price": "POST - Price a single option",
            "/api/price-chain": "POST - Price multiple options",
            "/api/compare": "POST - Compare pricing methods",
            "/api/implied-vol": "POST - Calculate implied volatility",
            "/api/vol-surface": "GET - Generate volatility surface",
            "/api/greeks": "POST - Calculate option Greeks",
            "/api/health": "GET - Health check"
        }
    })

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "pricing_manager_available": pricing_manager is not None
    })

@app.route('/api/price', methods=['POST'])
def price_option():
    """Price a single option"""
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['option', 'market', 'method']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        option_data = data['option']
        market_data = data['market']
        method = data['method']

        start_time = time.time()

        if pricing_manager:
            # Create option and market objects
            option = OptionContract(
                underlying=option_data.get('underlying', 'STOCK'),
                strike_price=float(option_data['strike_price']),
                option_type=OptionType.CALL if option_data['option_type'].lower() == 'call' else OptionType.PUT,
                option_style=OptionStyle.EUROPEAN if option_data['option_style'].lower() == 'european' else OptionStyle.AMERICAN,
                time_to_expiration=float(option_data['time_to_expiration'])
            )

            market = MarketData(
                underlying_price=float(market_data['underlying_price']),
                risk_free_rate=float(market_data['risk_free_rate']),
                dividend_yield=float(market_data.get('dividend_yield', 0.0)),
                volatility=float(market_data['volatility']),
                time_to_expiration=float(option_data['time_to_expiration'])
            )

            # Map method string to enum
            method_map = {
                'black_scholes': PricingMethod.BLACK_SCHOLES,
                'monte_carlo': PricingMethod.MONTE_CARLO,
                'quantlib': PricingMethod.QUANTLIB
            }

            pricing_method = method_map.get(method.lower())
            if not pricing_method:
                return jsonify({"error": f"Unsupported pricing method: {method}"}), 400

            # Price the option
            result = pricing_manager.price_option(option, market, pricing_method)

            return jsonify({
                "success": True,
                "result": result.to_dict(),
                "request_id": str(int(time.time() * 1000000))
            })

        else:
            # Mock response
            calc_time = (time.time() - start_time) * 1000
            mock_price = np.random.uniform(5.0, 15.0)
            result = create_mock_result(mock_price, method)
            result["calculation_time"] = calc_time

            return jsonify({
                "success": True,
                "result": result,
                "request_id": str(int(time.time() * 1000000)),
                "note": "Mock result - pricing engine not available"
            })

    except Exception as e:
        logger.error(f"Error pricing option: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/price-chain', methods=['POST'])
def price_option_chain():
    """Price multiple options"""
    try:
        data = request.get_json()

        options_data = data.get('options', [])
        market_data = data.get('market', SAMPLE_MARKET_DATA)
        method = data.get('method', 'black_scholes')

        results = []

        for option_data in options_data:
            # Price each option
            single_request = {
                'option': option_data,
                'market': market_data,
                'method': method
            }

            # Simulate individual pricing call
            if pricing_manager:
                # Real implementation would go here
                pass

            # Mock result for now
            mock_price = np.random.uniform(3.0, 20.0)
            result = create_mock_result(mock_price, method)
            results.append({
                "option": option_data,
                "result": result
            })

        return jsonify({
            "success": True,
            "results": results,
            "total_options": len(results)
        })

    except Exception as e:
        logger.error(f"Error pricing option chain: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare_methods():
    """Compare results from different pricing methods"""
    try:
        data = request.get_json()

        option_data = data['option']
        market_data = data['market']
        methods = data.get('methods', ['black_scholes', 'monte_carlo', 'quantlib'])

        comparison_results = {}

        for method in methods:
            start_time = time.time()

            if pricing_manager:
                # Real implementation
                try:
                    option = OptionContract(
                        underlying=option_data.get('underlying', 'STOCK'),
                        strike_price=float(option_data['strike_price']),
                        option_type=OptionType.CALL if option_data['option_type'].lower() == 'call' else OptionType.PUT,
                        option_style=OptionStyle.EUROPEAN if option_data['option_style'].lower() == 'european' else OptionStyle.AMERICAN,
                        time_to_expiration=float(option_data['time_to_expiration'])
                    )

                    market = MarketData(
                        underlying_price=float(market_data['underlying_price']),
                        risk_free_rate=float(market_data['risk_free_rate']),
                        dividend_yield=float(market_data.get('dividend_yield', 0.0)),
                        volatility=float(market_data['volatility']),
                        time_to_expiration=float(option_data['time_to_expiration'])
                    )

                    # This would call the real pricing engine
                    # result = pricing_manager.price_option(option, market, method)
                    # comparison_results[method] = result.to_dict()

                except Exception as e:
                    logger.error(f"Error with method {method}: {e}")

            # Mock results for demonstration
            calc_time = (time.time() - start_time) * 1000
            base_price = 10.0

            # Add some variation between methods
            if method == 'monte_carlo':
                price = base_price + np.random.normal(0, 0.1)
            elif method == 'black_scholes':
                price = base_price
            else:
                price = base_price + np.random.normal(0, 0.05)

            result = create_mock_result(price, method)
            result["calculation_time"] = calc_time
            comparison_results[method] = result

        # Calculate comparison statistics
        prices = [result["option_price"] for result in comparison_results.values()]
        stats = {
            "mean_price": np.mean(prices),
            "std_price": np.std(prices),
            "min_price": np.min(prices),
            "max_price": np.max(prices),
            "price_range": np.max(prices) - np.min(prices)
        }

        return jsonify({
            "success": True,
            "results": comparison_results,
            "statistics": stats
        })

    except Exception as e:
        logger.error(f"Error comparing methods: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/implied-vol', methods=['POST'])
def calculate_implied_volatility():
    """Calculate implied volatility"""
    try:
        data = request.get_json()

        option_data = data['option']
        market_data = data['market']
        market_price = float(data['market_price'])

        start_time = time.time()

        if pricing_manager:
            # Real implementation would go here
            pass

        # Mock implied volatility calculation
        calc_time = (time.time() - start_time) * 1000
        implied_vol = np.random.uniform(0.15, 0.35)

        return jsonify({
            "success": True,
            "implied_volatility": implied_vol,
            "calculation_time": calc_time,
            "market_price": market_price,
            "iterations": np.random.randint(5, 25)
        })

    except Exception as e:
        logger.error(f"Error calculating implied volatility: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/vol-surface')
def generate_volatility_surface():
    """Generate volatility surface"""
    try:
        # Parameters from query string
        underlying_price = float(request.args.get('underlying_price', 100))
        min_strike_ratio = float(request.args.get('min_strike_ratio', 0.8))
        max_strike_ratio = float(request.args.get('max_strike_ratio', 1.2))
        min_expiry = float(request.args.get('min_expiry', 0.1))
        max_expiry = float(request.args.get('max_expiry', 2.0))

        # Generate strikes and expiries
        strikes = np.linspace(
            underlying_price * min_strike_ratio,
            underlying_price * max_strike_ratio,
            10
        )
        expiries = np.linspace(min_expiry, max_expiry, 8)

        surface_data = []

        for expiry in expiries:
            for strike in strikes:
                # Mock volatility surface with realistic shape
                moneyness = strike / underlying_price
                time_effect = 0.02 * np.sqrt(expiry)
                skew_effect = 0.05 * (1.1 - moneyness)
                base_vol = 0.2

                implied_vol = base_vol + time_effect + skew_effect + np.random.normal(0, 0.01)
                implied_vol = max(0.05, min(implied_vol, 1.0))  # Reasonable bounds

                surface_data.append({
                    "strike": round(strike, 2),
                    "expiry": round(expiry, 3),
                    "moneyness": round(moneyness, 4),
                    "implied_vol": round(implied_vol, 4),
                    "vol_percentage": round(implied_vol * 100, 2)
                })

        return jsonify({
            "success": True,
            "surface_data": surface_data,
            "parameters": {
                "underlying_price": underlying_price,
                "strike_range": [strikes[0], strikes[-1]],
                "expiry_range": [expiries[0], expiries[-1]],
                "data_points": len(surface_data)
            }
        })

    except Exception as e:
        logger.error(f"Error generating volatility surface: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/greeks', methods=['POST'])
def calculate_greeks():
    """Calculate option Greeks"""
    try:
        data = request.get_json()

        option_data = data['option']
        market_data = data['market']

        start_time = time.time()

        # Mock Greeks calculation
        calc_time = (time.time() - start_time) * 1000

        # Generate realistic Greeks based on option type and moneyness
        strike = float(option_data['strike_price'])
        underlying = float(market_data['underlying_price'])
        moneyness = strike / underlying
        is_call = option_data['option_type'].lower() == 'call'

        if is_call:
            delta = max(0, min(1, 0.5 + 0.3 * (1 - moneyness)))
        else:
            delta = max(-1, min(0, -0.5 - 0.3 * (moneyness - 1)))

        greeks = {
            "delta": round(delta, 4),
            "gamma": round(np.random.uniform(0.005, 0.05), 4),
            "theta": round(np.random.uniform(-0.05, -0.005), 4),
            "vega": round(np.random.uniform(0.05, 0.3), 4),
            "rho": round(np.random.uniform(0.02, 0.15), 4)
        }

        return jsonify({
            "success": True,
            "greeks": greeks,
            "calculation_time": calc_time,
            "moneyness": round(moneyness, 4)
        })

    except Exception as e:
        logger.error(f"Error calculating Greeks: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/market-scenarios', methods=['POST'])
def run_market_scenarios():
    """Run option pricing under different market scenarios"""
    try:
        data = request.get_json()

        base_option = data['option']
        base_market = data['market']
        scenarios = data.get('scenarios', [])

        results = []

        for scenario in scenarios:
            # Apply scenario changes to market data
            scenario_market = base_market.copy()
            scenario_market.update(scenario.get('market_changes', {}))

            # Mock pricing for each scenario
            mock_price = np.random.uniform(5.0, 20.0)
            result = create_mock_result(mock_price, 'black_scholes')

            results.append({
                "scenario_name": scenario.get('name', 'Unnamed Scenario'),
                "market_data": scenario_market,
                "pricing_result": result
            })

        return jsonify({
            "success": True,
            "base_market": base_market,
            "scenario_results": results
        })

    except Exception as e:
        logger.error(f"Error running market scenarios: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance-benchmark', methods=['POST'])
def performance_benchmark():
    """Benchmark different pricing methods"""
    try:
        data = request.get_json()

        iterations = data.get('iterations', 100)
        methods = data.get('methods', ['black_scholes', 'monte_carlo'])

        benchmark_results = {}

        for method in methods:
            times = []

            for _ in range(iterations):
                start_time = time.time()

                # Mock pricing calculation
                if method == 'monte_carlo':
                    time.sleep(np.random.uniform(0.001, 0.01))  # Simulate MC calculation
                else:
                    time.sleep(np.random.uniform(0.0001, 0.001))  # Simulate analytical calculation

                calc_time = (time.time() - start_time) * 1000
                times.append(calc_time)

            benchmark_results[method] = {
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
                "median_time": np.median(times),
                "iterations": iterations
            }

        return jsonify({
            "success": True,
            "benchmark_results": benchmark_results,
            "test_parameters": {
                "iterations": iterations,
                "methods_tested": methods
            }
        })

    except Exception as e:
        logger.error(f"Error running performance benchmark: {e}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    print(f"Starting High-Frequency Options Pricing Engine API on port {port}")
    print(f"Debug mode: {debug}")

    app.run(host='0.0.0.0', port=port, debug=debug)
