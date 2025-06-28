# High Frequency Options Pricing Engine

# High-Performance Options Pricing Engine

A robust, multi-language framework for high-frequency financial derivative pricing and risk analysis. This project combines a high-performance C++ core with a flexible Python/QuantLib integration layer, exposed via a Flask REST API and consumed by an interactive React frontend.

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Core Features](#core-features)
3.  [System Architecture](#system-architecture)
    -   [Architectural Flow](#architectural-flow)
4.  [Technology Stack](#technology-stack)
5.  [Mathematical Foundations](#mathematical-foundations)
6.  [Installation and Setup](#installation-and-setup)
    -   [Prerequisites](#prerequisites)
    -   [Build & Execution Steps](#build--execution-steps)
7.  [Usage Guide](#usage-guide)
    -   [Python Engine Interaction](#python-engine-interaction)
    -   [Frontend API Integration](#frontend-api-integration)
8.  [API Reference](#api-reference)
    -   [Endpoints](#endpoints)
    -   [Example Request Body](#example-request-body)
9.  [Performance Characteristics](#performance-characteristics)
10. [Testing and Validation](#testing-and-validation)
11. [Contributing](#contributing)
12. [License](#license)

## Project Overview

This project provides a full-stack solution for the computationally intensive task of pricing financial options. It is designed to serve as a production-ready engine for quantitative analysts, developers, and risk managers who require speed, accuracy, and flexibility. The system decouples the high-performance computational core from the user-facing application, resulting in a scalable and maintainable architecture.

The primary goals of this implementation are:
-   **Performance**: Achieve microsecond-level pricing for standard options.
-   **Accuracy**: Provide reliable and validated implementations of industry-standard pricing models.
-   **Extensibility**: Allow for the easy addition of new financial models and features.
-   **Usability**: Offer multiple interfaces for interaction, from direct Python scripting to a full-featured web application.

## Core Features

-   **Multi-Model Support**: Implements three distinct pricing methodologies:
    -   **Black-Scholes**: Analytical closed-form solution for European options.
    -   **Monte Carlo Simulation**: For pricing exotic and path-dependent options using Geometric Brownian Motion.
    -   **Finite Difference Method**: Numerical solution of the Black-Scholes PDE, suitable for American options with early-exercise features.
-   **Real-time Risk Sensitivities (Greeks)**: Calculation of Delta, Gamma, Vega, Theta, and Rho for comprehensive risk management.
-   **Advanced Calibration Tools**: Includes implied volatility calibration from market prices and generation of complete volatility surfaces.
-   **Full-Stack Interface**: A complete solution from the C++ core to a React-based interactive web dashboard.

## System Architecture

The engine is built on a layered, polyglot architecture that leverages the strengths of each technology. This design ensures that computationally intensive tasks are handled by a low-level language, while flexibility and web integration are managed by high-level frameworks.

![Architecture Diagram](Architecture.png)

### Architectural Flow

1.  **Frontend (React)**: The user interacts with the web application, inputting option parameters and market data.
2.  **API Gateway (Flask)**: The React client sends a JSON payload to the Flask API. The API validates the request and acts as the entry point to the backend system.
3.  **Integration Layer (Python)**: The Python wrapper receives the data, constructs appropriate `OptionContract` and `MarketData` objects, and interfaces with the QuantLib library for high-level financial logic or delegates the request to the C++ core.
4.  **Computational Core (C++)**: The C++ engine performs the pricing or risk calculation at high speed and returns the result to the Python layer.
5.  **Response**: The result is serialized back into a JSON object by Flask and returned to the React client for display.

## Technology Stack

-   **Core Engine**: C++17, Boost, QuantLib
-   **Backend & Integration**: Python 3.x, Flask, NumPy, Pandas, SciPy, QuantLib-Python
-   **Frontend**: React, JavaScript
-   **Build System**: CMake, g++

## Mathematical Foundations

The engine provides implementations of the following core financial models:

-   **Black-Scholes Formula**: `C(S, K, T, r, σ)`
-   **Monte Carlo Simulation (Geometric Brownian Motion)**: `dS(t) = S(t) * ((r - q)dt + σ dW(t))`
-   **Finite Difference Method (Black-Scholes PDE)**: `∂V/∂t + ½σ²S²∂²V/∂S² + (r-q)S∂V/∂S - rV = 0`

## Installation and Setup

### Prerequisites

-   A C++17 compliant compiler (e.g., GCC, Clang)
-   CMake (version 3.10+)
-   Python (version 3.8+) and pip
-   Boost C++ Libraries
-   QuantLib C++ Library and Python bindings

### Build & Execution Steps

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/hft-options-engine.git
    cd hft-options-engine
    ```

2.  **Install Dependencies**:
    -   **System Libraries (Debian/Ubuntu Example)**:
        ```bash
        sudo apt-get update
        sudo apt-get install g++ cmake libboost-all-dev
        # Follow official documentation to install QuantLib
        ```
    -   **Python Packages**:
        ```bash
        pip install -r requirements.txt
        ```

3.  **Compile the Core Engine**:
    ```bash
    g++ -std=c++17 -O3 -Wall OptionsPricingEngine.cpp -o pricing_engine -lQuantLib
    ```

4.  **Run the Backend Server**:
    ```bash
    python flask_api.py
    ```
    The API will be available at `http://127.0.0.1:5000`.

5.  **Launch the Frontend**:
    Open the `index.html` file in a web browser. The application will connect to the local API server.

## Usage Guide

### Python Engine Interaction
The `python_pricing_engine.py` module provides a high-level manager for direct interaction.

```python
from python_pricing_engine import HFPricingManager, MarketData, OptionContract, PricingMethod

# 1. Define market conditions and the option contract
market = MarketData(
    underlying_price=200.0,
    risk_free_rate=0.03,
    volatility=0.25,
    time_to_expiration=0.5
)
option = OptionContract(
    strike_price=205.0,
    option_type="call",
    option_style="european",
    time_to_expiration=0.5
)

# 2. Initialize the manager and price the option
manager = HFPricingManager()
result = manager.price_option(option, market, PricingMethod.BLACK_SCHOLES)

print(f"Option Price: {result.option_price:.4f}")
print(f"Delta: {result.greeks.delta:.4f}")
print(f"Calculation Time (ms): {result.calculation_time:.4f}")
