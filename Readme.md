# Strategy Factory

**Strategy Factory** is a web application designed to let you code, test and deploy your trading bots easily — all from your browser!
<p align="center">
  <img src="./images/home.png" alt="Interface" style="width: 60%; max-width: 600px;">
</p>

## What can you do with Strategy Factory?

### 🤖 Trading Strategy Coding
Code your strategies directly in the browser, with visual tools to analyze their behavior.

🔹 *Built on top of [Backtesting.py](https://github.com/kernc/backtesting.py)* → If you already use this library, it will feel familiar.

![Demo GIF](images/run_strategy.gif)

### ⚡ Mass Strategy Execution
Run your bots across multiple tickers and timeframes from your broker.

![Demo GIF](images/backtests.gif)

### 🧪 Strategy Testing
Put your strategy to the test with the following methods:
 - Monte Carlo
 - Random Test
 - Luck Test
 - T-Test
 - Correlation Test

![Demo GIF](images/bot_tests.gif)

### 🚀 Deploy your bot
Put your bot to run in real time!

![Demo GIF](images/deploy_bot.gif)

## ⚙️ Installation

### Requirements
- [Python 3.12.8](https://www.python.org/downloads/release/python-3128/)
- [MetaTrader5](https://www.metatrader5.com/en)

### Steps
- Log in with your broker account on MetaTrader5
- Create and activate a virtual environment
- Run `python install_dependencies.py`
- Launch the app with `python app/main.py`

## Notes and Recommendations
- In some cases, brokers don’t use "pure" ticker names (e.g., EURUSD might be EURUSDm). In such cases, the app might fail.

- Currently, Strategy Factory assumes the account base currency used for operations is **USD**.

- The core of the application is built on the [Backtesting.py](https://github.com/kernc/backtesting.py) library, but to implement some extra features, a [fork](https://github.com/SaidJacobo/backtesting.py) of the project was made.

## Docs
📕 https://saidjacobo.github.io/strategy_factory_core/app/backbone.html

## Tutorial (only in spanish)
📺 https://www.youtube.com/playlist?list=PLIS81qU4XbMc8n5pinieZsrb4K4hhutjw

## 🤝 Contributing

Contributions are welcome and greatly appreciated! 🎉 Whether you want to report a bug, request a new feature, improve the documentation, or submit code improvements — you are more than welcome.

### How to Contribute

1. **Fork this repository** — Click the `Fork` button at the top right.
2. **Clone your fork**
```bash
git clone https://github.com/your-username/strategy_factory.git
```
3. **Create a new branch**
```bash
git checkout -b feature/your-feature-name
```
4. **Make your changes**  
Implement bug fixes, new features, or improvements.
5. **Commit your changes**
```bash
git commit -m "Add: description of your change"
```
6. **Push to your fork**
```bash
git push origin feature/your-feature-name
```
7. **Open a Pull Request** — Go to the repository and click `New pull request`.

### Guidelines

- Follow the existing code style and structure.
- Write clear and descriptive commit messages.
- If possible, include tests for your changes.
- Document any new features or behaviors.

### Discussions and Issues

If you're not sure where to start, check the [Issues](https://github.com/SaidJacobo/strategy_factory_core/issues) tab for open bugs or feature requests. You can also start a discussion if you have ideas or questions.

---

## ⭐ Support the Project

If you find this project useful, consider giving it a ⭐ star on GitHub — it really helps!


## Contact
📧 saidjacobo06@gmail.com
