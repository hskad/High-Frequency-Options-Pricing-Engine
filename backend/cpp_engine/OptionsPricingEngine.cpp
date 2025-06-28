
#include "OptionsPricingEngine.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>

namespace OptionsPricing {

// Monte Carlo Engine Implementation
MonteCarloEngine::MonteCarloEngine(int paths, int timeSteps, unsigned int seed)
    : numPaths_(paths), numTimeSteps_(timeSteps), seed_(seed),
      useAntithetic_(false), useControlVariates_(false) {}

PricingResult MonteCarloEngine::price(const OptionContract& option, 
                                     const MarketData& market) {
    auto start = std::chrono::high_resolution_clock::now();

    PricingResult result;
    result.method = "Monte Carlo";

    // Generate price paths
    std::vector<double> finalPrices = generatePricePaths(market, numPaths_, numTimeSteps_);

    // Calculate payoffs
    std::vector<double> payoffs;
    payoffs.reserve(numPaths_);

    for (double finalPrice : finalPrices) {
        payoffs.push_back(calculatePayoff(option, finalPrice));
    }

    // Calculate discounted expected value
    double sumPayoffs = std::accumulate(payoffs.begin(), payoffs.end(), 0.0);
    double averagePayoff = sumPayoffs / numPaths_;
    double discountFactor = std::exp(-market.riskFreeRate * option.timeToExpiration);

    result.optionPrice = averagePayoff * discountFactor;

    // Calculate standard error for confidence intervals
    double variance = 0.0;
    for (double payoff : payoffs) {
        double diff = payoff - averagePayoff;
        variance += diff * diff;
    }
    variance /= (numPaths_ - 1);
    double standardError = std::sqrt(variance / numPaths_) * discountFactor;

    // Estimate Greeks using finite differences (simplified)
    const double bump = 0.01;
    MarketData bumpedMarket = market;

    // Delta calculation
    bumpedMarket.underlyingPrice = market.underlyingPrice + bump;
    std::vector<double> bumpedPrices = generatePricePaths(bumpedMarket, numPaths_ / 10, numTimeSteps_);
    double bumpedSum = 0.0;
    for (double price : bumpedPrices) {
        bumpedSum += calculatePayoff(option, price);
    }
    double bumpedPrice = (bumpedSum / bumpedPrices.size()) * std::exp(-market.riskFreeRate * option.timeToExpiration);
    result.greeks.delta = (bumpedPrice - result.optionPrice) / bump;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    result.calculationTime = duration.count();
    result.convergenceSteps = numPaths_;

    return result;
}

std::vector<double> MonteCarloEngine::generatePricePaths(const MarketData& market, 
                                                        int numPaths, int timeSteps) {
    std::vector<double> finalPrices;
    finalPrices.reserve(numPaths);

    std::mt19937 generator(seed_);
    std::normal_distribution<double> normal(0.0, 1.0);

    double dt = market.timeToExpiration / timeSteps;
    double drift = (market.riskFreeRate - market.dividendYield - 0.5 * market.volatility * market.volatility) * dt;
    double diffusion = market.volatility * std::sqrt(dt);

    for (int path = 0; path < numPaths; ++path) {
        double currentPrice = market.underlyingPrice;

        for (int step = 0; step < timeSteps; ++step) {
            double randomShock = normal(generator);
            currentPrice *= std::exp(drift + diffusion * randomShock);
        }

        finalPrices.push_back(currentPrice);

        // Antithetic variate if enabled
        if (useAntithetic_ && path < numPaths / 2) {
            currentPrice = market.underlyingPrice;
            for (int step = 0; step < timeSteps; ++step) {
                double randomShock = -normal(generator);  // Antithetic
                currentPrice *= std::exp(drift + diffusion * randomShock);
            }
            finalPrices.push_back(currentPrice);
            ++path; // Skip one iteration since we added two paths
        }
    }

    return finalPrices;
}

double MonteCarloEngine::calculatePayoff(const OptionContract& option, double finalPrice) {
    switch (option.type) {
        case OptionType::CALL:
            return std::max(finalPrice - option.strikePrice, 0.0);
        case OptionType::PUT:
            return std::max(option.strikePrice - finalPrice, 0.0);
        default:
            return 0.0;
    }
}

void MonteCarloEngine::setParameters(const std::map<std::string, double>& params) {
    if (params.find("numPaths") != params.end()) {
        numPaths_ = static_cast<int>(params.at("numPaths"));
    }
    if (params.find("timeSteps") != params.end()) {
        numTimeSteps_ = static_cast<int>(params.at("timeSteps"));
    }
    if (params.find("seed") != params.end()) {
        seed_ = static_cast<unsigned int>(params.at("seed"));
    }
}

// Black-Scholes Engine Implementation
PricingResult BlackScholesEngine::price(const OptionContract& option, 
                                       const MarketData& market) {
    auto start = std::chrono::high_resolution_clock::now();

    PricingResult result;
    result.method = "Black-Scholes Analytical";

    if (option.style == OptionStyle::AMERICAN) {
        // For American options, use a simple approximation or return error
        result.optionPrice = 0.0;
        return result;
    }

    double S = market.underlyingPrice;
    double K = option.strikePrice;
    double r = market.riskFreeRate;
    double q = market.dividendYield;
    double sigma = market.volatility;
    double T = option.timeToExpiration;

    double d1_val = d1(S, K, r, q, sigma, T);
    double d2_val = d2(S, K, r, q, sigma, T);

    double Nd1 = normalCDF(d1_val);
    double Nd2 = normalCDF(d2_val);
    double discountFactor = std::exp(-r * T);
    double dividendDiscount = std::exp(-q * T);

    switch (option.type) {
        case OptionType::CALL:
            result.optionPrice = S * dividendDiscount * Nd1 - K * discountFactor * Nd2;
            break;
        case OptionType::PUT:
            result.optionPrice = K * discountFactor * (1.0 - Nd2) - S * dividendDiscount * (1.0 - Nd1);
            break;
    }

    // Calculate Greeks
    result.greeks = calculateGreeks(option, market);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    result.calculationTime = duration.count();
    result.convergenceSteps = 1; // Analytical solution

    return result;
}

double BlackScholesEngine::d1(double S, double K, double r, double q, 
                             double sigma, double T) {
    return (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
}

double BlackScholesEngine::d2(double S, double K, double r, double q, 
                             double sigma, double T) {
    return d1(S, K, r, q, sigma, T) - sigma * std::sqrt(T);
}

double BlackScholesEngine::normalCDF(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

double BlackScholesEngine::normalPDF(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

Greeks BlackScholesEngine::calculateGreeks(const OptionContract& option, 
                                          const MarketData& market) {
    Greeks greeks;

    double S = market.underlyingPrice;
    double K = option.strikePrice;
    double r = market.riskFreeRate;
    double q = market.dividendYield;
    double sigma = market.volatility;
    double T = option.timeToExpiration;

    double d1_val = d1(S, K, r, q, sigma, T);
    double d2_val = d2(S, K, r, q, sigma, T);
    double Nd1 = normalCDF(d1_val);
    double nd1 = normalPDF(d1_val);
    double Nd2 = normalCDF(d2_val);

    double discountFactor = std::exp(-r * T);
    double dividendDiscount = std::exp(-q * T);

    // Delta
    switch (option.type) {
        case OptionType::CALL:
            greeks.delta = dividendDiscount * Nd1;
            break;
        case OptionType::PUT:
            greeks.delta = dividendDiscount * (Nd1 - 1.0);
            break;
    }

    // Gamma (same for calls and puts)
    greeks.gamma = (dividendDiscount * nd1) / (S * sigma * std::sqrt(T));

    // Theta
    double theta1 = -(S * dividendDiscount * nd1 * sigma) / (2.0 * std::sqrt(T));
    double theta2 = r * K * discountFactor;
    double theta3 = q * S * dividendDiscount;

    switch (option.type) {
        case OptionType::CALL:
            greeks.theta = (theta1 - theta2 * Nd2 + theta3 * Nd1) / 365.0;
            break;
        case OptionType::PUT:
            greeks.theta = (theta1 + theta2 * (1.0 - Nd2) - theta3 * (1.0 - Nd1)) / 365.0;
            break;
    }

    // Vega (same for calls and puts)
    greeks.vega = S * dividendDiscount * nd1 * std::sqrt(T) / 100.0;

    // Rho
    switch (option.type) {
        case OptionType::CALL:
            greeks.rho = K * T * discountFactor * Nd2 / 100.0;
            break;
        case OptionType::PUT:
            greeks.rho = -K * T * discountFactor * (1.0 - Nd2) / 100.0;
            break;
    }

    return greeks;
}

void BlackScholesEngine::setParameters(const std::map<std::string, double>& params) {
    // Black-Scholes has no configurable parameters
}

// HF Pricing Manager Implementation
HFPricingManager::HFPricingManager() : cachingEnabled_(false) {
    engines_[PricingMethod::MONTE_CARLO] = std::make_unique<MonteCarloEngine>();
    engines_[PricingMethod::FINITE_DIFFERENCE] = std::make_unique<FiniteDifferenceEngine>();
    engines_[PricingMethod::BLACK_SCHOLES] = std::make_unique<BlackScholesEngine>();
}

PricingResult HFPricingManager::priceOption(const OptionContract& option,
                                           const MarketData& market,
                                           PricingMethod method) {
    auto it = engines_.find(method);
    if (it != engines_.end()) {
        return it->second->price(option, market);
    }

    // Default to Black-Scholes if method not found
    return engines_[PricingMethod::BLACK_SCHOLES]->price(option, market);
}

double HFPricingManager::calibrateImpliedVolatility(const OptionContract& option,
                                                   const MarketData& market,
                                                   double marketPrice,
                                                   double tolerance) {
    // Newton-Raphson method for implied volatility calculation
    MarketData tempMarket = market;
    double vol = 0.2; // Initial guess

    for (int iter = 0; iter < 100; ++iter) {
        tempMarket.volatility = vol;
        PricingResult result = engines_[PricingMethod::BLACK_SCHOLES]->price(option, tempMarket);

        double price = result.optionPrice;
        double vega = result.greeks.vega * 100.0; // Convert from percentage

        double diff = price - marketPrice;

        if (std::abs(diff) < tolerance) {
            return vol;
        }

        if (std::abs(vega) < 1e-10) {
            break; // Avoid division by zero
        }

        vol = vol - diff / vega;

        // Ensure volatility stays positive
        vol = std::max(vol, 0.001);
        vol = std::min(vol, 5.0); // Cap at 500%
    }

    return vol;
}

// Math Utilities Implementation
namespace MathUtils {
    double normalCDF(double x) {
        return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    }

    double normalPDF(double x) {
        return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
    }

    RandomGenerator::RandomGenerator(unsigned int seed) 
        : seed_(seed), hasSpare_(false), spare_(0.0) {}

    double RandomGenerator::normal() {
        if (hasSpare_) {
            hasSpare_ = false;
            return spare_;
        }

        hasSpare_ = true;
        static std::mt19937 generator(seed_);
        static std::uniform_real_distribution<double> uniform(0.0, 1.0);

        double u1 = uniform(generator);
        double u2 = uniform(generator);

        double magnitude = std::sqrt(-2.0 * std::log(u1));
        spare_ = magnitude * std::cos(2.0 * M_PI * u2);

        return magnitude * std::sin(2.0 * M_PI * u2);
    }
}

} // namespace OptionsPricing
