#!/bin/bash

# climaXtreme setup script

set -e

echo "Setting up climaXtreme development environment..."

# Check if Python 3.9+ is available
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+' || echo "0.0")
required_version="3.9"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
    echo "Error: Python 3.9 or higher is required. Found: $python_version"
    exit 1
fi

echo "✓ Python version check passed"

# Check if Java is installed (required for PySpark)
if ! command -v java &> /dev/null; then
    echo "Warning: Java not found. Installing OpenJDK 11..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y openjdk-11-jdk
    elif command -v yum &> /dev/null; then
        sudo yum install -y java-11-openjdk-devel
    elif command -v brew &> /dev/null; then
        brew install openjdk@11
    else
        echo "Please install Java 11 manually"
        exit 1
    fi
fi

echo "✓ Java check passed"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "✓ Virtual environment created"

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install package in development mode
pip install -e .

echo "✓ Python dependencies installed"

# Create necessary directories
echo "Creating directory structure..."
mkdir -p data/{raw,processed,output}
mkdir -p logs
mkdir -p models
mkdir -p notebooks

echo "✓ Directory structure created"

# Install pre-commit hooks
if command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit hooks..."
    pre-commit install
    echo "✓ Pre-commit hooks installed"
else
    echo "pre-commit not found, skipping hook installation"
fi

# Set JAVA_HOME if not already set
if [ -z "$JAVA_HOME" ]; then
    if [ -d "/usr/lib/jvm/java-11-openjdk-amd64" ]; then
        echo "export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64" >> ~/.bashrc
        export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
    elif [ -d "/usr/lib/jvm/java-11-openjdk" ]; then
        echo "export JAVA_HOME=/usr/lib/jvm/java-11-openjdk" >> ~/.bashrc
        export JAVA_HOME=/usr/lib/jvm/java-11-openjdk
    fi
    echo "✓ JAVA_HOME configured"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "To get started:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run data ingestion: climaxtreme ingest"
echo "3. Preprocess data: climaxtreme preprocess"
echo "4. Launch dashboard: climaxtreme dashboard"
echo ""
echo "For more information, see README.md"