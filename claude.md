# Claude Context – Trading System

## 🧠 General Role

You are working on a production-grade algorithmic trading platform.

This system:
- Supports strategy development, backtesting, and live trading
- Uses a custom fork of backtesting.py as the core engine
- Integrates with MetaTrader 5 for live execution
- Is designed to maintain strict consistency between backtesting and live trading

You must prioritize:
1. Correctness over speed
2. Consistency between backtest and live trading
3. Avoiding duplication of logic
4. Preserving system architecture

---

## 🏗️ Architecture Rules (STRICT)

### Layer responsibilities

- Controllers (FastAPI + Jinja):
  - Only handle HTTP requests and responses
  - Must NOT contain business logic
  - Must ONLY call services

- Services:
  - Contain all business logic
  - Can call other services if needed
  - Must NOT access the database directly

- Data Layer (SQLAlchemy):
  - Responsible ONLY for persistence
  - No business logic allowed

### Forbidden

- Controllers accessing data layer directly ❌
- Business logic inside controllers ❌
- Database access outside repositories ❌

---

## ⚙️ Strategy System (CORE)

### Definition

All strategies:
- Must inherit from `StrategyFactory`
- `StrategyFactory` inherits from backtesting.py `Strategy`

Strategies include required properties:
- pip_value
- minimum_lot
- maximum_lot
- contract_volume
- volume_step
- risk
- metatrader_name
- ticker
- live
- timezone
- minimum_fraction

### Responsibilities

A strategy is responsible for:
- Signal generation (buy/sell/close)
- Position sizing (risk, lot calculation)

### Constraints

- Strategies MUST be broker-agnostic
- Strategies MUST NOT contain MT5-specific logic
- Strategies MUST behave identically in backtest and live

---

## 🔁 Backtest vs Live Consistency (CRITICAL)

The system is designed to ensure:

- Same strategy code runs in backtest and live
- Same execution semantics
- Same SL/TP behavior
- Same sizing logic

### Allowed differences

- Minor differences due to:
  - Latency
  - Real market slippage

### Forbidden

- Diverging logic between backtest and live ❌
- Conditional behavior that changes results ❌

---

## 🔴 Execution Model

- Strategies call:
  - `buy()`
  - `sell()`
  - `close()`

- Behavior:
  - In backtest → handled by engine
  - In live (`live=True`) → routed to MetaTrader 5

This routing MUST remain transparent and consistent.

### Important

- Do NOT modify execution flow without explicit request
- Do NOT introduce alternative execution paths

---

## 📉 Orders

Current state:
- Market orders only

Planned:
- Limit / Stop orders (not yet implemented)

### SL/TP

- Must be defined at order creation
- No dynamic SL/TP management (except closing positions)

---

## 🔌 Live Trading Integration

- Uses MetaTrader 5
- Execution triggered via cron (external scheduler)
- System is stateless

### Key rules

- No in-memory state for positions
- Each execution must infer state from current data

### Live classes

- LiveTrade
- LivePosition

These mirror backtesting.py structures and MUST remain consistent.

---

## 🧱 Portfolio System

- A portfolio = collection of strategies
- Metrics are calculated from combined equity curves

### Capital model

- Each strategy:
  - Has independent risk (typically % of equity)
  - Operates independently

- Strategies:
  - Can open positions simultaneously
  - Do NOT compete for capital in current implementation

---

## 🚫 Critical Rules (DO NOT VIOLATE)

### 1. No logic duplication

- NEVER duplicate:
  - Backtesting logic
  - Execution logic
  - Sizing logic

### 2. No arbitrary changes

- Do NOT modify existing behavior unless explicitly requested
- Do NOT refactor unrelated parts

### 3. Impact awareness

- Any change must consider:
  - Backtest impact
  - Live trading impact
  - Strategy compatibility

If unsure → ASK before proceeding

---

## ⚠️ Known Problem Areas

Be extremely careful with:

1. Strategy properties:
   - contract size
   - lot size
   - volume step

2. Cron execution timing:
   - Must align with candle close
   - Avoid pre/post-candle execution errors

---

## 🧩 Coding Standards

- Language: Python
- Always use type hints
- Always include docstrings
- Prefer explicit and readable code over clever code

---

## 🧪 Testing

- Do NOT generate tests unless explicitly requested

---

## 🔒 Modification Policy (VERY IMPORTANT)

You must be conservative.

Before making changes that affect:
- Execution logic
- Strategy behavior
- Live trading

You MUST:
1. Explain the impact
2. Ask for confirmation

---

## 🧠 Guiding Principle

This is a trading system handling real money.

Bad code here does not just fail — it loses capital.

Act accordingly.