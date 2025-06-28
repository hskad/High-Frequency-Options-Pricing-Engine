# High-Frequency Options Pricing Engine Implementation Guide

## Project Overview

This project implements a comprehensive high-frequency options pricing engine using multiple technologies to create a fast, accurate, and scalable solution for pricing financial derivatives. The system combines C++ for computational performance, Python for flexibility, Flask for API services, and React for the user interface.

## Architecture Components

### 1. C++ Core Engine (`OptionsPricingEngine.hpp` / `.cpp`)

The C++ core provides maximum computational speed for high-frequency scenarios:

**Key Features:**
- **Monte Carlo Engine**: Implements geometric Brownian motion simulation with variance reduction techniques
- **Finite Difference Engine**: Solves Black-Scholes PDE using explicit, implicit, and Crank-Nicolson schemes
- **Black-Scholes Analytical Engine**: Closed-form solutions for European options
- **Greeks Calculation**: Real-time sensitivity analysis (Delta, Gamma, Theta, Vega, Rho)
- **High-Performance**: Optimized for microsecond-level pricing

**Technical Specifications:**
```cpp
// Core pricing interface
class PricingEngine {
    virtual PricingResult price(const OptionContract& option, 
                               const MarketData& market) = 0;
};

// Monte Carlo with variance reduction
class MonteCarloEngine : public PricingEngine {
    // Up to 1,000,000 simulation paths
    // Antithetic variates and control variates
    // Multi-threaded execution support
};

// Finite difference with multiple schemes
class FiniteDifferenceEngine : public PricingEngine {
    // Explicit, Implicit, Crank-Nicolson schemes
    // Adaptive grid sizing
    // Boundary condition handling
};
```

### 2. Python Integration Layer (`python_pricing_engine.py`)

Python wrapper provides flexibility and QuantLib integration:

**Key Features:**
- **QuantLib Integration**: Professional-grade financial library support
- **Data Structures**: Type-safe option contracts and market data
- **Calibration Tools**: Implied volatility calculation and volatility surface generation
- **Performance Monitoring**: Detailed timing and convergence analysis

**Example Usage:**
```python
# Create market data and option contract
market = MarketData(
    underlying_price=100.0,
    risk_free_rate=0.05,
    volatility=0.2,
    time_to_expiration=1.0
)

option = OptionContract(
    underlying="AAPL",
    strike_price=100.0,
    option_type=OptionType.CALL,
    option_style=OptionStyle.EUROPEAN,
    time_to_expiration=1.0
)

# Price using different methods
manager = HFPricingManager()
result = manager.price_option(option, market, PricingMethod.MONTE_CARLO)
```

### 3. Flask API Backend (`flask_api.py`)

RESTful API provides web service interface:

**Endpoints:**
- `POST /api/price` - Price single option
- `POST /api/price-chain` - Price multiple options
- `POST /api/compare` - Compare pricing methods
- `POST /api/implied-vol` - Calculate implied volatility
- `GET /api/vol-surface` - Generate volatility surface
- `POST /api/greeks` - Calculate option Greeks
- `POST /api/performance-benchmark` - Performance testing

**Example API Call:**
```json
POST /api/price
{
  "option": {
    "strike_price": 100,
    "option_type": "call",
    "option_style": "european",
    "time_to_expiration": 1.0
  },
  "market": {
    "underlying_price": 100,
    "risk_free_rate": 0.05,
    "volatility": 0.2
  },
  "method": "black_scholes"
}
```

### 4. React Frontend Web Application

Interactive web interface for option pricing:

**Features:**
- **Multiple Pricing Methods**: Monte Carlo, Finite Difference, Black-Scholes
- **Real-time Calculations**: Instant price updates as parameters change
- **Visualization**: Price paths, volatility surfaces, Greeks plots
- **Comparison Tools**: Side-by-side method comparison
- **Export Capabilities**: CSV and JSON data export

## Mathematical Implementation

### Black-Scholes Formula

European call option price:
```
C = S₀ × e^(-q×T) × N(d₁) - K × e^(-r×T) × N(d₂)

where:
d₁ = [ln(S₀/K) + (r - q + σ²/2) × T] / (σ × √T)
d₂ = d₁ - σ × √T
```

### Monte Carlo Simulation

Geometric Brownian motion:
```
S(t+dt) = S(t) × exp[(r - q - σ²/2) × dt + σ × √dt × ε]

where ε ~ N(0,1)
```

### Finite Difference Method

Black-Scholes PDE:
```
∂V/∂t + ½σ²S²∂²V/∂S² + (r-q)S∂V/∂S - rV = 0
```

Discretized using Crank-Nicolson scheme for stability.

## Performance Characteristics

### Computational Speed

| Method | Typical Time | Accuracy | Use Case |
|--------|-------------|----------|----------|
| Black-Scholes | < 1ms | Exact | European options |
| Monte Carlo | 10-100ms | Convergent | Exotic options |
| Finite Difference | 5-50ms | High | American options |

### Memory Usage

- **Monte Carlo**: O(N × M) where N = paths, M = time steps
- **Finite Difference**: O(N × M) where N = price grid, M = time grid
- **Black-Scholes**: O(1) constant memory

### Scalability

- **Single Option**: Microsecond pricing
- **Option Chain**: Linear scaling with number of strikes
- **Real-time**: Capable of 1000+ prices per second

## Deployment Instructions

### Prerequisites

```bash
# C++ dependencies
sudo apt-get install g++ cmake libboost-all-dev

# QuantLib installation
wget https://dl.bintray.com/quantlib/releases/QuantLib-1.29.tar.gz
tar xzf QuantLib-1.29.tar.gz
cd QuantLib-1.29
./configure
make && sudo make install

# Python dependencies
pip install numpy pandas scipy Flask Flask-CORS QuantLib-Python
```

### Compilation

```bash
# Compile C++ engine
g++ -std=c++17 -O3 -Wall OptionsPricingEngine.cpp -o pricing_engine \
    -lQuantLib -lboost_system -lboost_date_time

# Run Python tests
python python_pricing_engine.py

# Start Flask API
python flask_api.py
```

### Web Application

```bash
# Serve the React application
# The application is already deployed and accessible via the provided URL
```

## Usage Examples

### Basic Option Pricing

```python
# Import required modules
from python_pricing_engine import *

# Define market conditions
market = MarketData(
    underlying_price=100.0,
    risk_free_rate=0.05,
    dividend_yield=0.02,
    volatility=0.25,
    time_to_expiration=0.5
)

# Create option contract
option = OptionContract(
    underlying="STOCK",
    strike_price=105.0,
    option_type=OptionType.CALL,
    option_style=OptionStyle.EUROPEAN,
    time_to_expiration=0.5
)

# Price the option
manager = HFPricingManager()
result = manager.price_option(option, market, PricingMethod.BLACK_SCHOLES)

print(f"Option Price: ${result.option_price:.4f}")
print(f"Delta: {result.greeks.delta:.4f}")
print(f"Calculation Time: {result.calculation_time:.2f}ms")
```

### Volatility Surface Generation

```python
# Generate strikes around current price
strikes = np.linspace(80, 120, 9)
expiries = np.linspace(0.1, 2.0, 8)

# Create volatility surface
surface = manager.generate_volatility_surface(strikes, expiries, market)
print(surface.head())
```

### Performance Comparison

```python
# Compare all methods
results = manager.compare_methods(option, market)

for method, result in results.items():
    print(f"{method}: ${result.option_price:.4f} ({result.calculation_time:.2f}ms)")
```

## API Integration

### JavaScript Frontend Integration

```javascript
// Price option via API
async function priceOption(optionData, marketData, method) {
    const response = await fetch('/api/price', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            option: optionData,
            market: marketData,
            method: method
        })
    });
    
    const result = await response.json();
    return result;
}

// Example usage
const option = {
    strike_price: 100,
    option_type: 'call',
    option_style: 'european',
    time_to_expiration: 1.0
};

const market = {
    underlying_price: 100,
    risk_free_rate: 0.05,
    volatility: 0.2
};

priceOption(option, market, 'black_scholes')
    .then(result => console.log(result));
```

## Advanced Features

### Implied Volatility Calibration

The system can calibrate implied volatility from market prices using Newton-Raphson iteration:

```python
# Calibrate implied volatility
market_price = 8.50
implied_vol = manager.calibrate_implied_volatility(
    option, market, market_price, tolerance=1e-6
)
print(f"Implied Volatility: {implied_vol:.4f}")
```

### Risk Management

Calculate portfolio-level Greeks:

```python
# Price option chain
option_chain = [
    OptionContract("STOCK", 95, OptionType.CALL, OptionStyle.EUROPEAN, 0.5),
    OptionContract("STOCK", 100, OptionType.CALL, OptionStyle.EUROPEAN, 0.5),
    OptionContract("STOCK", 105, OptionType.CALL, OptionStyle.EUROPEAN, 0.5)
]

# Calculate portfolio Greeks
portfolio_delta = 0
portfolio_gamma = 0

for option in option_chain:
    result = manager.price_option(option, market, PricingMethod.BLACK_SCHOLES)
    portfolio_delta += result.greeks.delta
    portfolio_gamma += result.greeks.gamma

print(f"Portfolio Delta: {portfolio_delta:.4f}")
print(f"Portfolio Gamma: {portfolio_gamma:.4f}")
```

## Testing and Validation

### Unit Tests

```python
# Test Black-Scholes against known values
def test_black_scholes():
    # Known test case
    option = OptionContract("TEST", 100, OptionType.CALL, OptionStyle.EUROPEAN, 1.0)
    market = MarketData(100, 0.05, 0.0, 0.2, 1.0)
    
    result = manager.price_option(option, market, PricingMethod.BLACK_SCHOLES)
    expected_price = 10.45  # Known theoretical value
    
    assert abs(result.option_price - expected_price) < 0.01
```

### Performance Benchmarks

```python
# Benchmark pricing speed
import time

def benchmark_pricing():
    times = []
    for _ in range(1000):
        start = time.time()
        result = manager.price_option(option, market, PricingMethod.BLACK_SCHOLES)
        times.append((time.time() - start) * 1000)
    
    print(f"Average time: {np.mean(times):.2f}ms")
    print(f"95th percentile: {np.percentile(times, 95):.2f}ms")
```

## Troubleshooting

### Common Issues

1. **QuantLib Installation**: Ensure QuantLib is properly compiled with Python bindings
2. **Memory Usage**: Monitor memory for large Monte Carlo simulations
3. **Numerical Stability**: Check for extreme parameter values that may cause instability
4. **API Timeouts**: Set appropriate timeout values for long-running calculations

### Performance Optimization

1. **Compiler Flags**: Use `-O3 -march=native` for maximum C++ performance
2. **Threading**: Enable OpenMP for parallel Monte Carlo paths
3. **Caching**: Implement result caching for repeated calculations
4. **Memory Pool**: Use memory pools for frequent allocations

This implementation provides a production-ready foundation for high-frequency options pricing with extensibility for additional features and optimizations.