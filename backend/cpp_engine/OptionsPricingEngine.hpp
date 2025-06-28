
#ifndef OPTIONS_PRICING_ENGINE_H
#define OPTIONS_PRICING_ENGINE_H

#include <vector>
#include <string>
#include <memory>
#include <functional>

// Forward declarations
class QuantLib::Date;
class QuantLib::Calendar;

namespace OptionsPricing {

    // Option types
    enum class OptionType { CALL, PUT };
    enum class OptionStyle { EUROPEAN, AMERICAN };
    enum class PricingMethod { MONTE_CARLO, FINITE_DIFFERENCE, BLACK_SCHOLES };

    // Market data structure
    struct MarketData {
        double underlyingPrice;
        double riskFreeRate;
        double dividendYield;
        double volatility;
        double timeToExpiration;
    };

    // Option contract specification
    struct OptionContract {
        std::string underlying;
        double strikePrice;
        OptionType type;
        OptionStyle style;
        double timeToExpiration;
        
        OptionContract(const std::string& und, double strike, OptionType opt_type, 
                      OptionStyle opt_style, double tte)
            : underlying(und), strikePrice(strike), type(opt_type), 
              style(opt_style), timeToExpiration(tte) {}
    };

    // Greeks structure
    struct Greeks {
        double delta;    // Price sensitivity to underlying
        double gamma;    // Delta sensitivity to underlying
        double theta;    // Time decay
        double vega;     // Volatility sensitivity
        double rho;      // Interest rate sensitivity
        
        Greeks() : delta(0.0), gamma(0.0), theta(0.0), vega(0.0), rho(0.0) {}
    };

    // Pricing result
    struct PricingResult {
        double optionPrice;
        Greeks greeks;
        double impliedVolatility;
        double calculationTime;  // in microseconds
        int convergenceSteps;
        std::string method;
        
        PricingResult() : optionPrice(0.0), impliedVolatility(0.0), 
                         calculationTime(0.0), convergenceSteps(0) {}
    };

    // Base pricing engine interface
    class PricingEngine {
    public:
        virtual ~PricingEngine() = default;
        virtual PricingResult price(const OptionContract& option, 
                                   const MarketData& market) = 0;
        virtual void setParameters(const std::map<std::string, double>& params) = 0;
    };

    // Monte Carlo pricing engine
    class MonteCarloEngine : public PricingEngine {
    private:
        int numPaths_;
        int numTimeSteps_;
        unsigned int seed_;
        bool useAntithetic_;
        bool useControlVariates_;
        
    public:
        MonteCarloEngine(int paths = 100000, int timeSteps = 252, 
                        unsigned int seed = 12345);
        
        PricingResult price(const OptionContract& option, 
                           const MarketData& market) override;
        
        void setParameters(const std::map<std::string, double>& params) override;
        
        // Monte Carlo specific methods
        std::vector<double> generatePricePaths(const MarketData& market, 
                                              int numPaths, int timeSteps);
        double calculatePayoff(const OptionContract& option, double finalPrice);
        void enableVarianceReduction(bool antithetic, bool controlVariates);
    };

    // Finite Difference pricing engine
    class FiniteDifferenceEngine : public PricingEngine {
    public:
        enum class Scheme { EXPLICIT, IMPLICIT, CRANK_NICOLSON };
        
    private:
        int gridSize_;
        int timeSteps_;
        Scheme scheme_;
        double maxPrice_;
        double minPrice_;
        
    public:
        FiniteDifferenceEngine(int gridSize = 200, int timeSteps = 500, 
                             Scheme scheme = Scheme::CRANK_NICOLSON);
        
        PricingResult price(const OptionContract& option, 
                           const MarketData& market) override;
        
        void setParameters(const std::map<std::string, double>& params) override;
        
        // FD specific methods
        void setupGrid(const OptionContract& option, const MarketData& market);
        void applyBoundaryConditions(const OptionContract& option, 
                                   const MarketData& market);
        std::vector<std::vector<double>> solveBackward(const OptionContract& option, 
                                                      const MarketData& market);
    };

    // Black-Scholes analytical engine
    class BlackScholesEngine : public PricingEngine {
    public:
        BlackScholesEngine() = default;
        
        PricingResult price(const OptionContract& option, 
                           const MarketData& market) override;
        
        void setParameters(const std::map<std::string, double>& params) override;
        
        // Analytical methods
        static double d1(double S, double K, double r, double q, 
                        double sigma, double T);
        static double d2(double S, double K, double r, double q, 
                        double sigma, double T);
        static double normalCDF(double x);
        static double normalPDF(double x);
        
        // Greeks calculations
        static Greeks calculateGreeks(const OptionContract& option, 
                                    const MarketData& market);
    };

    // High-frequency pricing manager
    class HFPricingManager {
    private:
        std::map<PricingMethod, std::unique_ptr<PricingEngine>> engines_;
        std::vector<PricingResult> resultCache_;
        bool cachingEnabled_;
        
    public:
        HFPricingManager();
        ~HFPricingManager() = default;
        
        // Main pricing methods
        PricingResult priceOption(const OptionContract& option, 
                                 const MarketData& market, 
                                 PricingMethod method);
        
        std::vector<PricingResult> priceOptionChain(
            const std::vector<OptionContract>& options,
            const MarketData& market,
            PricingMethod method);
        
        // Batch pricing for high-frequency scenarios
        void priceBatch(const std::vector<OptionContract>& options,
                       const std::vector<MarketData>& marketData,
                       PricingMethod method,
                       std::function<void(const PricingResult&)> callback);
        
        // Performance optimization
        void enableCaching(bool enable) { cachingEnabled_ = enable; }
        void clearCache() { resultCache_.clear(); }
        
        // Calibration and validation
        double calibrateImpliedVolatility(const OptionContract& option,
                                        const MarketData& market,
                                        double marketPrice,
                                        double tolerance = 1e-6);
        
        // Utility methods
        static double impliedVolatility(const OptionContract& option,
                                      const MarketData& market,
                                      double marketPrice);
        
        static std::vector<double> generateVolatilitySurface(
            const std::vector<double>& strikes,
            const std::vector<double>& expiries,
            const MarketData& baseMarket);
    };

    // Mathematical utilities
    namespace MathUtils {
        double normalCDF(double x);
        double normalPDF(double x);
        double normalInverse(double p);
        
        // Random number generation
        class RandomGenerator {
        public:
            RandomGenerator(unsigned int seed = 12345);
            double normal();
            double uniform();
            void setSeed(unsigned int seed);
            
        private:
            unsigned int seed_;
            bool hasSpare_;
            double spare_;
        };
        
        // Numerical methods
        double bisection(std::function<double(double)> f, 
                        double a, double b, double tolerance = 1e-6);
        double newtonRaphson(std::function<double(double)> f,
                           std::function<double(double)> df,
                           double x0, double tolerance = 1e-6);
    }

} // namespace OptionsPricing

#endif // OPTIONS_PRICING_ENGINE_H
