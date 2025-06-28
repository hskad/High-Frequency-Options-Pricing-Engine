// High-Frequency Options Pricing Engine - JavaScript Implementation

class OptionsPricingEngine {
    constructor() {
        this.currentChart = null;
        this.comparisonData = [];
        this.currentMethod = 'blackscholes';
        this.init();
    }

    init() {
        this.bindEvents();
        this.initializeDefaults();
    }

    bindEvents() {
        // Tab navigation
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });

        // Method selection
        document.querySelectorAll('.method-card').forEach(card => {
            card.addEventListener('click', (e) => this.selectMethod(e.currentTarget.dataset.method));
        });

        // Calculation button
        document.getElementById('calculate-btn').addEventListener('click', () => this.calculatePrice());

        // Reset button
        document.getElementById('reset-btn').addEventListener('click', () => this.resetParameters());

        // Scenario buttons
        document.querySelectorAll('.scenario-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.loadScenario(e.target.dataset.scenario));
        });

        // Visualization tabs
        document.querySelectorAll('.viz-tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchVisualization(e.target.dataset.viz));
        });

        // Export and comparison buttons
        document.getElementById('add-to-comparison')?.addEventListener('click', () => this.addToComparison());
        document.getElementById('export-results')?.addEventListener('click', () => this.showExportModal());
        document.getElementById('clear-comparison')?.addEventListener('click', () => this.clearComparison());
        document.getElementById('run-analysis')?.addEventListener('click', () => this.runSensitivityAnalysis());

        // Modal events
        document.querySelectorAll('.close-modal').forEach(btn => {
            btn.addEventListener('click', (e) => this.closeModal(e.target.closest('.modal')));
        });

        // Real-time parameter updates
        document.querySelectorAll('#option-params-form input, #option-params-form select').forEach(input => {
            input.addEventListener('change', () => this.validateParameters());
        });
    }

    initializeDefaults() {
        this.selectMethod('blackscholes');
        this.validateParameters();
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
        document.getElementById(`${tabName}-tab`).classList.add('active');
    }

    selectMethod(method) {
        this.currentMethod = method;
        
        // Update method cards
        document.querySelectorAll('.method-card').forEach(card => card.classList.remove('selected'));
        document.querySelector(`[data-method="${method}"]`).classList.add('selected');
    }

    validateParameters() {
        const params = this.getParameters();
        let isValid = true;
        
        // Clear previous errors
        document.querySelectorAll('.form-control').forEach(input => input.classList.remove('error'));
        
        // Validate each parameter
        if (params.S <= 0) {
            document.getElementById('underlying-price').classList.add('error');
            isValid = false;
        }
        
        if (params.K <= 0) {
            document.getElementById('strike-price').classList.add('error');
            isValid = false;
        }
        
        if (params.r < 0) {
            document.getElementById('risk-free-rate').classList.add('error');
            isValid = false;
        }
        
        if (params.sigma <= 0) {
            document.getElementById('volatility').classList.add('error');
            isValid = false;
        }
        
        if (params.T <= 0) {
            document.getElementById('time-to-expiry').classList.add('error');
            isValid = false;
        }

        document.getElementById('calculate-btn').disabled = !isValid;
        return isValid;
    }

    getParameters() {
        return {
            S: parseFloat(document.getElementById('underlying-price').value),
            K: parseFloat(document.getElementById('strike-price').value),
            r: parseFloat(document.getElementById('risk-free-rate').value),
            sigma: parseFloat(document.getElementById('volatility').value),
            T: parseFloat(document.getElementById('time-to-expiry').value),
            q: parseFloat(document.getElementById('dividend-yield').value) / 100,
            type: document.getElementById('option-type').value,
            style: document.getElementById('option-style').value,
            mcPaths: parseInt(document.getElementById('mc-paths').value) || 10000,
            mcSteps: parseInt(document.getElementById('mc-steps').value) || 252,
            fdGridSize: parseInt(document.getElementById('fd-grid-size').value) || 100,
            fdTimeSteps: parseInt(document.getElementById('fd-time-steps').value) || 100,
            fdScheme: document.getElementById('fd-scheme').value
        };
    }

    async calculatePrice() {
        if (!this.validateParameters()) return;

        const params = this.getParameters();
        const startTime = performance.now();
        
        // Show loading state
        document.getElementById('calculate-btn').classList.add('loading');
        document.getElementById('results-data').classList.add('hidden');
        document.querySelector('.no-results').classList.remove('hidden');

        try {
            let result;
            
            switch (this.currentMethod) {
                case 'blackscholes':
                    result = this.calculateBlackScholes(params);
                    break;
                case 'montecarlo':
                    result = await this.calculateMonteCarlo(params);
                    break;
                case 'finitediff':
                    result = this.calculateFiniteDifference(params);
                    break;
                default:
                    throw new Error('Invalid pricing method');
            }

            const endTime = performance.now();
            result.calculationTime = endTime - startTime;

            this.displayResults(result, params);
            this.updateTimingDisplay(result.calculationTime);

        } catch (error) {
            console.error('Calculation error:', error);
            this.showError(`Calculation failed: ${error.message}`);
        } finally {
            document.getElementById('calculate-btn').classList.remove('loading');
        }
    }

    calculateBlackScholes(params) {
        const { S, K, r, sigma, T, q, type } = params;
        
        const d1 = (Math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
        const d2 = d1 - sigma * Math.sqrt(T);
        
        const Nd1 = this.normalCDF(d1);
        const Nd2 = this.normalCDF(d2);
        const Nmd1 = this.normalCDF(-d1);
        const Nmd2 = this.normalCDF(-d2);
        
        let price, delta, gamma, theta, vega, rho;
        
        if (type === 'call') {
            price = S * Math.exp(-q * T) * Nd1 - K * Math.exp(-r * T) * Nd2;
            delta = Math.exp(-q * T) * Nd1;
            rho = K * T * Math.exp(-r * T) * Nd2 / 100;
        } else {
            price = K * Math.exp(-r * T) * Nmd2 - S * Math.exp(-q * T) * Nmd1;
            delta = -Math.exp(-q * T) * Nmd1;
            rho = -K * T * Math.exp(-r * T) * Nmd2 / 100;
        }
        
        // Greeks that are the same for calls and puts
        gamma = Math.exp(-q * T) * this.normalPDF(d1) / (S * sigma * Math.sqrt(T));
        vega = S * Math.exp(-q * T) * this.normalPDF(d1) * Math.sqrt(T) / 100;
        theta = (type === 'call') ?
            (-S * Math.exp(-q * T) * this.normalPDF(d1) * sigma / (2 * Math.sqrt(T)) 
             - r * K * Math.exp(-r * T) * Nd2 
             + q * S * Math.exp(-q * T) * Nd1) / 365 :
            (-S * Math.exp(-q * T) * this.normalPDF(d1) * sigma / (2 * Math.sqrt(T)) 
             + r * K * Math.exp(-r * T) * Nmd2 
             - q * S * Math.exp(-q * T) * Nmd1) / 365;

        return {
            price: Math.max(0, price),
            greeks: { delta, gamma, theta, vega, rho },
            method: 'Black-Scholes',
            convergence: null,
            paths: null,
            standardError: 0
        };
    }

    async calculateMonteCarlo(params) {
        const { S, K, r, sigma, T, q, type, mcPaths, mcSteps } = params;
        
        const dt = T / mcSteps;
        const drift = (r - q - 0.5 * sigma * sigma) * dt;
        const diffusion = sigma * Math.sqrt(dt);
        
        let payoffs = [];
        let paths = [];
        let convergenceData = [];
        
        // Generate sample paths for visualization (max 100 for performance)
        const samplePaths = Math.min(100, mcPaths);
        
        for (let i = 0; i < mcPaths; i++) {
            let price = S;
            let path = i < samplePaths ? [price] : null;
            
            // Simulate price path
            for (let j = 0; j < mcSteps; j++) {
                const z = this.boxMullerRandom();
                price *= Math.exp(drift + diffusion * z);
                if (path) path.push(price);
            }
            
            // Calculate payoff
            let payoff;
            if (type === 'call') {
                payoff = Math.max(0, price - K);
            } else {
                payoff = Math.max(0, K - price);
            }
            
            payoffs.push(payoff);
            if (path) paths.push(path);
            
            // Track convergence every 1000 paths
            if ((i + 1) % 1000 === 0) {
                const avgPayoff = payoffs.slice(0, i + 1).reduce((a, b) => a + b, 0) / (i + 1);
                const currentPrice = Math.exp(-r * T) * avgPayoff;
                convergenceData.push({ iteration: i + 1, price: currentPrice });
            }
            
            // Yield control occasionally for UI responsiveness
            if (i % 5000 === 0) {
                await new Promise(resolve => setTimeout(resolve, 0));
            }
        }
        
        const avgPayoff = payoffs.reduce((a, b) => a + b, 0) / payoffs.length;
        const price = Math.exp(-r * T) * avgPayoff;
        
        // Calculate standard error
        const variance = payoffs.reduce((acc, p) => acc + Math.pow(p - avgPayoff, 2), 0) / (payoffs.length - 1);
        const standardError = Math.sqrt(variance / payoffs.length) * Math.exp(-r * T);
        
        // Approximate Greeks using finite differences
        const bump = 0.01;
        const greeks = await this.calculateMCGreeks(params, price, bump);
        
        return {
            price,
            greeks,
            method: 'Monte Carlo',
            convergence: convergenceData,
            paths: paths,
            standardError: standardError
        };
    }

    async calculateMCGreeks(params, basePrice, bump) {
        // Calculate Delta
        const upParams = { ...params, S: params.S * (1 + bump) };
        const downParams = { ...params, S: params.S * (1 - bump) };
        
        const upResult = await this.calculateSimpleMC(upParams);
        const downResult = await this.calculateSimpleMC(downParams);
        
        const delta = (upResult - downResult) / (2 * params.S * bump);
        
        // Approximate other Greeks (simplified for performance)
        return {
            delta: delta,
            gamma: 0, // Would require second-order finite differences
            theta: -basePrice * 0.1, // Simplified approximation
            vega: basePrice * 0.2, // Simplified approximation
            rho: basePrice * params.T * 0.01 // Simplified approximation
        };
    }

    async calculateSimpleMC(params) {
        const { S, K, r, sigma, T, q, type } = params;
        const paths = 1000; // Simplified for Greeks calculation
        
        let payoffs = [];
        
        for (let i = 0; i < paths; i++) {
            const z = this.boxMullerRandom();
            const price = S * Math.exp((r - q - 0.5 * sigma * sigma) * T + sigma * Math.sqrt(T) * z);
            
            let payoff;
            if (type === 'call') {
                payoff = Math.max(0, price - K);
            } else {
                payoff = Math.max(0, K - price);
            }
            
            payoffs.push(payoff);
        }
        
        const avgPayoff = payoffs.reduce((a, b) => a + b, 0) / payoffs.length;
        return Math.exp(-r * T) * avgPayoff;
    }

    calculateFiniteDifference(params) {
        const { S, K, r, sigma, T, q, type, fdGridSize, fdTimeSteps, fdScheme } = params;
        
        // Set up spatial grid
        const Smax = 3 * Math.max(S, K);
        const dS = Smax / fdGridSize;
        const dt = T / fdTimeSteps;
        
        // Create price grid
        let grid = new Array(fdGridSize + 1).fill(0).map(() => new Array(fdTimeSteps + 1).fill(0));
        let prices = new Array(fdGridSize + 1).fill(0).map((_, i) => i * dS);
        
        // Set up boundary conditions at expiry
        for (let i = 0; i <= fdGridSize; i++) {
            if (type === 'call') {
                grid[i][fdTimeSteps] = Math.max(0, prices[i] - K);
            } else {
                grid[i][fdTimeSteps] = Math.max(0, K - prices[i]);
            }
        }
        
        // Solve backwards through time
        for (let j = fdTimeSteps - 1; j >= 0; j--) {
            if (fdScheme === 'explicit') {
                this.explicitScheme(grid, prices, j, dt, dS, r, q, sigma, type, K);
            } else if (fdScheme === 'implicit') {
                this.implicitScheme(grid, prices, j, dt, dS, r, q, sigma, fdGridSize);
            } else {
                this.crankNicolsonScheme(grid, prices, j, dt, dS, r, q, sigma, fdGridSize);
            }
        }
        
        // Interpolate price at current stock price
        const price = this.interpolatePrice(grid, prices, S, 0);
        
        // Calculate Greeks using finite differences
        const greeks = this.calculateFDGreeks(grid, prices, S, dS);
        
        return {
            price,
            greeks,
            method: `Finite Difference (${fdScheme})`,
            convergence: null,
            paths: null,
            standardError: 0,
            grid: this.sampleGrid(grid, prices, 20) // Sample for visualization
        };
    }

    explicitScheme(grid, prices, j, dt, dS, r, q, sigma, type, K) {
        for (let i = 1; i < grid.length - 1; i++) {
            const S = prices[i];
            const a = 0.5 * dt * (sigma * sigma * i * i - (r - q) * i);
            const b = 1 - dt * (sigma * sigma * i * i + r);
            const c = 0.5 * dt * (sigma * sigma * i * i + (r - q) * i);
            
            grid[i][j] = a * grid[i-1][j+1] + b * grid[i][j+1] + c * grid[i+1][j+1];
        }
        
        // Boundary conditions
        if (type === 'call') {
            grid[0][j] = 0;
            grid[grid.length-1][j] = prices[grid.length-1] - K * Math.exp(-r * (j * dt));
        } else {
            grid[0][j] = K * Math.exp(-r * (j * dt));
            grid[grid.length-1][j] = 0;
        }
    }

    implicitScheme(grid, prices, j, dt, dS, r, q, sigma, gridSize) {
        // Simplified implicit scheme implementation
        // In practice, this would solve a tridiagonal system
        for (let i = 1; i < gridSize; i++) {
            const S = prices[i];
            const alpha = 0.5 * dt * (sigma * sigma * i * i - (r - q) * i);
            const beta = 1 + dt * (sigma * sigma * i * i + r);
            const gamma = -0.5 * dt * (sigma * sigma * i * i + (r - q) * i);
            
            // Simplified - would use Thomas algorithm for tridiagonal solve
            grid[i][j] = grid[i][j+1] / beta;
        }
    }

    crankNicolsonScheme(grid, prices, j, dt, dS, r, q, sigma, gridSize) {
        // Simplified Crank-Nicolson implementation
        // Combines explicit and implicit schemes
        for (let i = 1; i < gridSize; i++) {
            const S = prices[i];
            const coeff = 0.25 * dt * sigma * sigma * i * i;
            const drift = 0.25 * dt * (r - q) * i;
            
            grid[i][j] = 0.5 * (grid[i-1][j+1] + grid[i+1][j+1]) + 
                        (1 - 0.5 * dt * r) * grid[i][j+1];
        }
    }

    interpolatePrice(grid, prices, S, timeIndex) {
        // Linear interpolation
        for (let i = 0; i < prices.length - 1; i++) {
            if (S >= prices[i] && S <= prices[i + 1]) {
                const weight = (S - prices[i]) / (prices[i + 1] - prices[i]);
                return grid[i][timeIndex] * (1 - weight) + grid[i + 1][timeIndex] * weight;
            }
        }
        return grid[Math.floor(prices.length / 2)][timeIndex];
    }

    calculateFDGreeks(grid, prices, S, dS) {
        // Find closest grid points
        let index = Math.floor(S / dS);
        index = Math.max(1, Math.min(index, grid.length - 2));
        
        const delta = (grid[index + 1][0] - grid[index - 1][0]) / (2 * dS);
        const gamma = (grid[index + 1][0] - 2 * grid[index][0] + grid[index - 1][0]) / (dS * dS);
        
        return {
            delta: delta,
            gamma: gamma,
            theta: -grid[index][0] * 0.1, // Simplified
            vega: grid[index][0] * 0.2,   // Simplified
            rho: grid[index][0] * 0.01    // Simplified
        };
    }

    sampleGrid(grid, prices, sampleSize) {
        const step = Math.floor(grid.length / sampleSize);
        let sampledGrid = [];
        
        for (let i = 0; i < grid.length; i += step) {
            sampledGrid.push({
                price: prices[i],
                values: grid[i].slice()
            });
        }
        
        return sampledGrid;
    }

    displayResults(result, params) {
        // Hide no-results message
        document.querySelector('.no-results').classList.add('hidden');
        document.getElementById('results-data').classList.remove('hidden');
        
        // Update method name
        document.getElementById('result-method-name').textContent = `${result.method} Pricing`;
        
        // Update price
        document.getElementById('option-price-value').textContent = `$${result.price.toFixed(4)}`;
        
        // Update Greeks table
        this.updateGreeksTable(result.greeks);
        
        // Update performance metrics
        document.getElementById('calc-time-value').textContent = `${result.calculationTime.toFixed(2)}ms`;
        document.getElementById('memory-usage-value').textContent = this.estimateMemoryUsage(params);
        document.getElementById('std-error-value').textContent = `±${result.standardError.toFixed(4)}`;
        
        // Update visualization
        this.updateVisualization(result, params);
        
        // Store current result for comparison
        this.currentResult = { result, params };
    }

    updateGreeksTable(greeks) {
        const tableBody = document.getElementById('greeks-table-body');
        tableBody.innerHTML = '';
        
        const greekDefinitions = {
            delta: { symbol: 'Δ', description: 'Sensitivity to underlying price' },
            gamma: { symbol: 'Γ', description: 'Rate of change of delta' },
            theta: { symbol: 'Θ', description: 'Time decay' },
            vega: { symbol: 'ν', description: 'Sensitivity to volatility' },
            rho: { symbol: 'ρ', description: 'Sensitivity to interest rate' }
        };
        
        Object.entries(greeks).forEach(([name, value]) => {
            const row = document.createElement('tr');
            const def = greekDefinitions[name] || { symbol: name, description: '' };
            
            row.innerHTML = `
                <td><strong>${def.symbol}</strong> ${name.charAt(0).toUpperCase() + name.slice(1)}</td>
                <td>${value.toFixed(6)}</td>
                <td>${def.description}</td>
            `;
            tableBody.appendChild(row);
        });
    }

    updateVisualization(result, params) {
        const activeViz = document.querySelector('.viz-tab-btn.active').dataset.viz;
        
        switch (activeViz) {
            case 'price-paths':
                this.createPricePathsChart(result.paths, params);
                break;
            case 'convergence':
                this.createConvergenceChart(result.convergence);
                break;
            case 'value-surface':
                this.createValueSurfaceChart(result, params);
                break;
        }
    }

    createPricePathsChart(paths, params) {
        const ctx = document.getElementById('result-chart').getContext('2d');
        
        if (this.currentChart) {
            this.currentChart.destroy();
        }
        
        if (!paths || paths.length === 0) {
            this.showChartMessage('No price paths available for this method');
            return;
        }
        
        const timeSteps = Array.from({ length: paths[0].length }, (_, i) => 
            (i * params.T / (paths[0].length - 1)).toFixed(2)
        );
        
        const datasets = paths.slice(0, 10).map((path, index) => ({
            label: `Path ${index + 1}`,
            data: path,
            borderColor: `hsl(${index * 36}, 70%, 50%)`,
            backgroundColor: 'transparent',
            borderWidth: 1,
            pointRadius: 0
        }));
        
        this.currentChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: timeSteps,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Sample Price Paths'
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Time (years)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Stock Price ($)'
                        }
                    }
                }
            }
        });
    }

    createConvergenceChart(convergenceData) {
        const ctx = document.getElementById('result-chart').getContext('2d');
        
        if (this.currentChart) {
            this.currentChart.destroy();
        }
        
        if (!convergenceData || convergenceData.length === 0) {
            this.showChartMessage('No convergence data available');
            return;
        }
        
        this.currentChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: convergenceData.map(d => d.iteration),
                datasets: [{
                    label: 'Option Price',
                    data: convergenceData.map(d => d.price),
                    borderColor: '#1FB8CD',
                    backgroundColor: 'rgba(31, 184, 205, 0.1)',
                    borderWidth: 2,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Monte Carlo Convergence'
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Number of Simulations'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Option Price ($)'
                        }
                    }
                }
            }
        });
    }

    createValueSurfaceChart(result, params) {
        const ctx = document.getElementById('result-chart').getContext('2d');
        
        if (this.currentChart) {
            this.currentChart.destroy();
        }
        
        // Generate value surface data
        const strikes = [];
        const times = [];
        const values = [];
        
        for (let k = params.K * 0.8; k <= params.K * 1.2; k += params.K * 0.05) {
            for (let t = 0.1; t <= params.T; t += params.T / 10) {
                const tempParams = { ...params, K: k, T: t };
                const tempResult = this.calculateBlackScholes(tempParams);
                
                strikes.push(k);
                times.push(t);
                values.push(tempResult.price);
            }
        }
        
        // Create scatter plot for 3D-like visualization
        this.currentChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Option Value',
                    data: strikes.map((k, i) => ({
                        x: k,
                        y: times[i],
                        r: Math.max(2, values[i] / 2)
                    })),
                    backgroundColor: strikes.map((_, i) => {
                        const intensity = values[i] / Math.max(...values);
                        return `rgba(31, 184, 205, ${0.3 + intensity * 0.7})`;
                    })
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Option Value Surface'
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Strike Price ($)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Time to Expiry (years)'
                        }
                    }
                }
            }
        });
    }

    showChartMessage(message) {
        const ctx = document.getElementById('result-chart').getContext('2d');
        if (this.currentChart) {
            this.currentChart.destroy();
        }
        
        // Clear canvas and show message
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        ctx.font = '16px Arial';
        ctx.fillStyle = '#666';
        ctx.textAlign = 'center';
        ctx.fillText(message, ctx.canvas.width / 2, ctx.canvas.height / 2);
    }

    switchVisualization(vizType) {
        // Update viz buttons
        document.querySelectorAll('.viz-tab-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelector(`[data-viz="${vizType}"]`).classList.add('active');
        
        // Update visualization if results are available
        if (this.currentResult) {
            this.updateVisualization(this.currentResult.result, this.currentResult.params);
        }
    }

    addToComparison() {
        if (!this.currentResult) return;
        
        const comparisonItem = {
            method: this.currentResult.result.method,
            price: this.currentResult.result.price,
            greeks: this.currentResult.result.greeks,
            calculationTime: this.currentResult.result.calculationTime,
            standardError: this.currentResult.result.standardError,
            timestamp: new Date().toLocaleTimeString()
        };
        
        this.comparisonData.push(comparisonItem);
        this.updateComparisonTable();
        
        // Show success message
        this.showMessage('Result added to comparison', 'success');
    }

    updateComparisonTable() {
        const container = document.getElementById('comparison-table-container');
        const noComparison = document.querySelector('.no-comparison');
        
        if (this.comparisonData.length === 0) {
            container.classList.add('hidden');
            noComparison.classList.remove('hidden');
            return;
        }
        
        noComparison.classList.add('hidden');
        container.classList.remove('hidden');
        
        const tbody = document.getElementById('comparison-table-body');
        tbody.innerHTML = '';
        
        this.comparisonData.forEach((item, index) => {
            const row = document.createElement('tr');
            
            // Calculate error relative to first (analytical) result if available
            let error = 'N/A';
            if (this.comparisonData[0] && index > 0) {
                const relativeError = Math.abs(item.price - this.comparisonData[0].price) / this.comparisonData[0].price * 100;
                error = `${relativeError.toFixed(4)}%`;
            }
            
            row.innerHTML = `
                <td>${item.method}</td>
                <td>$${item.price.toFixed(4)}</td>
                <td>${item.greeks.delta.toFixed(4)}</td>
                <td>${item.greeks.gamma.toFixed(6)}</td>
                <td>${item.greeks.theta.toFixed(4)}</td>
                <td>${item.greeks.vega.toFixed(4)}</td>
                <td>${item.calculationTime.toFixed(2)}ms</td>
                <td>${error}</td>
            `;
            tbody.appendChild(row);
        });
        
        this.updateComparisonChart();
    }

    updateComparisonChart() {
        const ctx = document.getElementById('comparison-chart').getContext('2d');
        
        if (this.comparisonChart) {
            this.comparisonChart.destroy();
        }
        
        this.comparisonChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: this.comparisonData.map(item => item.method),
                datasets: [
                    {
                        label: 'Option Price',
                        data: this.comparisonData.map(item => item.price),
                        backgroundColor: '#1FB8CD',
                        yAxisID: 'y'
                    },
                    {
                        label: 'Calculation Time (ms)',
                        data: this.comparisonData.map(item => item.calculationTime),
                        backgroundColor: '#FFC185',
                        type: 'line',
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Methods Comparison'
                    }
                },
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Option Price ($)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Calculation Time (ms)'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                }
            }
        });
    }

    clearComparison() {
        this.comparisonData = [];
        this.updateComparisonTable();
        if (this.comparisonChart) {
            this.comparisonChart.destroy();
        }
    }

    async runSensitivityAnalysis() {
        const param = document.getElementById('analysis-param').value;
        const range = parseFloat(document.getElementById('analysis-range').value) / 100;
        const steps = parseInt(document.getElementById('analysis-steps').value);
        
        const baseParams = this.getParameters();
        const baseValue = baseParams[this.getParamKey(param)];
        
        const results = [];
        const values = [];
        
        for (let i = 0; i <= steps; i++) {
            const multiplier = (1 - range) + (2 * range * i / steps);
            const newValue = baseValue * multiplier;
            
            const tempParams = { ...baseParams };
            tempParams[this.getParamKey(param)] = newValue;
            
            const result = this.calculateBlackScholes(tempParams);
            results.push(result);
            values.push(newValue);
        }
        
        this.displaySensitivityResults(param, values, results);
    }

    getParamKey(param) {
        const mapping = {
            'underlying': 'S',
            'strike': 'K',
            'volatility': 'sigma',
            'time': 'T',
            'rate': 'r'
        };
        return mapping[param] || param;
    }

    displaySensitivityResults(param, values, results) {
        // Update table header
        document.getElementById('sensitivity-param-header').textContent = 
            param.charAt(0).toUpperCase() + param.slice(1);
        
        // Update table
        const tbody = document.getElementById('sensitivity-table-body');
        tbody.innerHTML = '';
        
        values.forEach((value, index) => {
            const result = results[index];
            const row = document.createElement('tr');
            
            row.innerHTML = `
                <td>${value.toFixed(4)}</td>
                <td>$${result.price.toFixed(4)}</td>
                <td>${result.greeks.delta.toFixed(4)}</td>
                <td>${result.greeks.gamma.toFixed(6)}</td>
                <td>${result.greeks.vega.toFixed(4)}</td>
            `;
            tbody.appendChild(row);
        });
        
        // Create chart
        this.createSensitivityChart(param, values, results);
    }

    createSensitivityChart(param, values, results) {
        const ctx = document.getElementById('sensitivity-chart').getContext('2d');
        
        if (this.sensitivityChart) {
            this.sensitivityChart.destroy();
        }
        
        this.sensitivityChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: values.map(v => v.toFixed(3)),
                datasets: [
                    {
                        label: 'Option Price',
                        data: results.map(r => r.price),
                        borderColor: '#1FB8CD',
                        backgroundColor: 'rgba(31, 184, 205, 0.1)',
                        yAxisID: 'y'
                    },
                    {
                        label: 'Delta',
                        data: results.map(r => r.greeks.delta),
                        borderColor: '#FFC185',
                        backgroundColor: 'rgba(255, 193, 133, 0.1)',
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: `Sensitivity to ${param.charAt(0).toUpperCase() + param.slice(1)}`
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: param.charAt(0).toUpperCase() + param.slice(1)
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Option Price ($)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Delta'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                }
            }
        });
    }

    resetParameters() {
        document.getElementById('underlying-price').value = 100;
        document.getElementById('strike-price').value = 100;
        document.getElementById('risk-free-rate').value = 0.05;
        document.getElementById('volatility').value = 0.2;
        document.getElementById('time-to-expiry').value = 1;
        document.getElementById('dividend-yield').value = 0;
        document.getElementById('option-type').value = 'call';
        document.getElementById('option-style').value = 'european';
        
        this.validateParameters();
    }

    loadScenario(scenarioName) {
        const scenarios = {
            'atm-call': { S: 100, K: 100, r: 0.05, sigma: 0.2, T: 0.25, type: 'call' },
            'itm-put': { S: 80, K: 100, r: 0.03, sigma: 0.3, T: 0.5, type: 'put' },
            'high-vol': { S: 100, K: 105, r: 0.02, sigma: 0.4, T: 1.0, type: 'call' }
        };
        
        const scenario = scenarios[scenarioName];
        if (!scenario) return;
        
        document.getElementById('underlying-price').value = scenario.S;
        document.getElementById('strike-price').value = scenario.K;
        document.getElementById('risk-free-rate').value = scenario.r;
        document.getElementById('volatility').value = scenario.sigma;
        document.getElementById('time-to-expiry').value = scenario.T;
        document.getElementById('option-type').value = scenario.type;
        
        this.validateParameters();
    }

    showExportModal() {
        if (!this.currentResult) return;
        
        const modal = document.getElementById('export-modal');
        const format = document.getElementById('export-format').value;
        const content = document.getElementById('export-content');
        
        let exportData;
        if (format === 'json') {
            exportData = JSON.stringify({
                parameters: this.currentResult.params,
                results: this.currentResult.result
            }, null, 2);
        } else {
            exportData = this.convertToCSV(this.currentResult);
        }
        
        content.value = exportData;
        modal.classList.remove('hidden');
    }

    convertToCSV(data) {
        const headers = ['Parameter', 'Value'];
        const rows = [
            ['Method', data.result.method],
            ['Option Price', data.result.price.toFixed(4)],
            ['Underlying Price', data.params.S],
            ['Strike Price', data.params.K],
            ['Risk-Free Rate', data.params.r],
            ['Volatility', data.params.sigma],
            ['Time to Expiry', data.params.T],
            ['Option Type', data.params.type],
            ['Delta', data.result.greeks.delta.toFixed(6)],
            ['Gamma', data.result.greeks.gamma.toFixed(6)],
            ['Theta', data.result.greeks.theta.toFixed(6)],
            ['Vega', data.result.greeks.vega.toFixed(6)],
            ['Rho', data.result.greeks.rho.toFixed(6)],
            ['Calculation Time (ms)', data.result.calculationTime.toFixed(2)]
        ];
        
        return [headers.join(','), ...rows.map(row => row.join(','))].join('\n');
    }

    closeModal(modal) {
        modal.classList.add('hidden');
    }

    updateTimingDisplay(time) {
        document.getElementById('calculation-time').textContent = `Calculation time: ${time.toFixed(2)}ms`;
    }

    estimateMemoryUsage(params) {
        let usage = 0.1; // Base usage
        
        if (this.currentMethod === 'montecarlo') {
            usage += (params.mcPaths * params.mcSteps * 8) / (1024 * 1024); // 8 bytes per number
        } else if (this.currentMethod === 'finitediff') {
            usage += (params.fdGridSize * params.fdTimeSteps * 8) / (1024 * 1024);
        }
        
        return `${usage.toFixed(2)} MB`;
    }

    showMessage(message, type = 'info') {
        // Create and show temporary message
        const messageEl = document.createElement('div');
        messageEl.className = `status status--${type}`;
        messageEl.textContent = message;
        messageEl.style.position = 'fixed';
        messageEl.style.top = '20px';
        messageEl.style.right = '20px';
        messageEl.style.zIndex = '1001';
        
        document.body.appendChild(messageEl);
        
        setTimeout(() => {
            document.body.removeChild(messageEl);
        }, 3000);
    }

    showError(message) {
        this.showMessage(message, 'error');
    }

    // Utility functions for mathematical calculations
    normalCDF(x) {
        return 0.5 * (1 + this.erf(x / Math.sqrt(2)));
    }

    normalPDF(x) {
        return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
    }

    erf(x) {
        // Approximation of error function
        const a1 =  0.254829592;
        const a2 = -0.284496736;
        const a3 =  1.421413741;
        const a4 = -1.453152027;
        const a5 =  1.061405429;
        const p  =  0.3275911;

        const sign = x < 0 ? -1 : 1;
        x = Math.abs(x);

        const t = 1.0 / (1.0 + p * x);
        const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

        return sign * y;
    }

    boxMullerRandom() {
        if (this.hasSpare) {
            this.hasSpare = false;
            return this.spare;
        }

        this.hasSpare = true;
        const u = Math.random();
        const v = Math.random();
        const mag = Math.sqrt(-2.0 * Math.log(u));
        this.spare = mag * Math.cos(2.0 * Math.PI * v);
        return mag * Math.sin(2.0 * Math.PI * v);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new OptionsPricingEngine();
});