# LOTER-IA - Lottery Number Prediction with AI

A machine learning system that predicts lottery numbers using LSTM neural networks and web scraping of historical data from ONCE (Spain's National Organization for the Blind).

## 📋 Features

- **Web Scraping**: Automatically collects historical lottery results from the official website
- **Local Database**: SQLite database for efficient data storage and retrieval
- **Machine Learning**: LSTM neural networks powered by PyTorch for predictions
- **Interactive CLI**: User-friendly command-line interface with autocompletion
- **Multiple Games**: Supports 7 different lottery types (Cuponazo, Diario, Madre, Verano, Navidad, Padre, Sueldazo)

## 🚀 Installation

### Requirements
- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LOTER-IA.git
cd LOTER-IA
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Running the Application

```bash
python main.py
or
./run.bat
```

### Available Commands

- **populate**: Download all historical lottery data (from 1996 onwards)
- **update**: Update data for the current year only
- **train**: Train the prediction models
- **predict**: Generate predictions using trained models
- **help**: Show help message
- **exit**: Exit the program

### Quick Start

```bash
# 1. Download historical data (first time only)
populate

# 2. Train models for all lottery types
train
> all

# 3. Generate predictions
predict
> all
```



## 📊 Project Structure

```
LOTER-IA/
├── main.py                 # Main entry point
├── config.py               # Configuration (Database, CLI)
├── lottery.py              # Lottery data model
├── loteria.py              # ML model - NumberPredictor
├── helpers.py              # Helper functions for parsing
├── dbcontroller.py         # SQLite database controller
├── test_lottery.py         # Unit tests suite
├── requirements.txt        # Dependencies
├── .gitignore             # Git ignore rules
├── SECURITY_AUDIT.md      # Security audit report
└── data/
    ├── results.json       # Historical lottery data
    └── last_numbers.json  # Latest predictions
```

## 🔧 Main Dependencies

- **torch**: Deep learning framework
- **pandas**: Data analysis and manipulation
- **playwright**: Web scraping automation
- **prompt_toolkit**: Interactive CLI
- **pyfiglet**: ASCII art for banners

See `requirements.txt` for the complete list.

## 📈 Available Models

An independent model is trained for each lottery type:
- `lottery_number_predictor_cuponazo.pth`
- `lottery_number_predictor_diario.pth`
- `lottery_number_predictor_madre.pth`
- `lottery_number_predictor_navidad.pth`
- `lottery_number_predictor_padre.pth`
- `lottery_number_predictor_sueldazo.pth`
- `lottery_number_predictor_verano.pth`

## ⚖️ Disclaimer

**IMPORTANT**: This is an educational and research project. The predictions do NOT guarantee results in real lotteries. Lotteries are random events and cannot be predicted with certainty. Use this project only for Machine Learning learning purposes.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is free to use, share, and modify. However, you must give appropriate credit to the original author.

**Creative Commons Attribution (CC BY)**: You are free to use this project for any purpose, provided that you give credit to the original author.

## ⚠️ Maintenance Status

**This project is no longer actively maintained.** It is provided as-is for educational and research purposes. While bug reports and pull requests may be reviewed, there is no guarantee of timely responses or regular updates. Feel free to fork the project and maintain your own version if needed.

## 👨‍💻 Author

Developed as an educational project for Machine Learning and Web Scraping.

**Please cite or credit the original author when using this project.**

## 📞 Support

To report issues or suggestions, please open an issue on GitHub. Note that responses may be limited as this project is not actively maintained.

