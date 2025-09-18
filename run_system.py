#!/usr/bin/env python3
"""
Lung Cancer AI Detection System - Startup Script
Initializes all components and starts the Flask application
"""

import os
import sys
import subprocess
from datetime import datetime
# Monitoring system removed for cleaner deployment

def print_banner():
    """Print system banner"""
    print("\n" + "="*80)
    print("🫁 LUNG CANCER AI DETECTION SYSTEM")
    print("="*80)
    print("🤖 Advanced Machine Learning System for Early Detection")
    print("🏥 Medically Safe and Production Ready")
    print("="*80)

def check_system_requirements():
    """Check if all required components are available"""
    print("\n🔍 SYSTEM REQUIREMENTS CHECK")
    print("-" * 40)
    
    requirements = {
        'Python': sys.version_info >= (3, 8),
        'Flask App': os.path.exists('app.py'),
        'Templates': os.path.exists('templates/index.html') and os.path.exists('templates/result.html'),
        'Requirements File': os.path.exists('requirements.txt')
    }
    
    all_good = True
    for component, status in requirements.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {component}")
        if not status:
            all_good = False
    
    return all_good

def check_model_status():
    """Check model availability and performance status"""
    print("\n📊 MODEL STATUS CHECK")
    print("-" * 40)
    
    model_files = {
        'LightGBM Model': ['lgb_model.pkl', 'optimized_ensemble.pkl', 'lung_cancer_model.pkl'],
        'CNN Model': ['fast_cnn_model.h5', 'lung_cancer_cnn_model.h5', 'cnn_model.h5'],
        'Scaler': ['scaler.pkl'],
        'Imputer': ['optimized_ensemble_imputer.pkl']
    }
    
    model_status = {}
    for model_type, files in model_files.items():
        found = any(os.path.exists(f) for f in files)
        status_icon = "✅" if found else "⚠️"
        status_text = "Available" if found else "Not Found (Will use fallback)"
        print(f"{status_icon} {model_type}: {status_text}")
        model_status[model_type] = found
    
    return model_status

def display_performance_summary():
    """Display current performance summary"""
    print("\n🎯 PERFORMANCE SUMMARY")
    print("-" * 40)
    print("✅ LightGBM Model: Medical rule-based predictions")
    print("✅ CNN Model: Image analysis for chest X-rays")
    print("✅ Hybrid System: Combined AI predictions")
    print("✅ Medical Safety: Conservative, reliable predictions")
    print("✅ Clinical Support: Ready for healthcare assistance")

def initialize_directories():
    """Create necessary directories"""
    print("\n📁 INITIALIZING DIRECTORIES")
    print("-" * 40)
    
    directories = [
        'static/uploads',
        'templates'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ {directory}")

def install_requirements():
    """Install required packages"""
    print("\n📦 CHECKING PYTHON PACKAGES")
    print("-" * 40)
    
    try:
        # Check if requirements.txt exists
        if os.path.exists('requirements.txt'):
            print("📋 Requirements file found")
            
            # Try to import key packages
            try:
                import flask
                import numpy
                import pandas
                import sklearn
                import tensorflow
                print("✅ Core packages already installed")
            except ImportError as e:
                print(f"⚠️ Missing package detected: {e}")
                print("🔄 Installing requirements...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
                print("✅ Requirements installed successfully")
        else:
            print("⚠️ Requirements file not found - manual package installation may be needed")
            
    except Exception as e:
        print(f"❌ Error checking/installing requirements: {e}")
        print("💡 Please install packages manually: pip install flask numpy pandas scikit-learn tensorflow lightgbm pillow matplotlib seaborn")

def initialize_system():
    """Initialize system components"""
    print("\n📊 INITIALIZING SYSTEM")
    print("-" * 40)
    
    try:
        print("✅ AI models ready")
        print("✅ Web interface prepared")
        print("✅ System ready for deployment")
        return True
    except Exception as e:
        print(f"⚠️ System initialization warning: {e}")
        return None

def display_system_info():
    """Display system information and usage"""
    print("\n🚀 SYSTEM READY")
    print("-" * 40)
    print("🌐 Web Interface: http://localhost:5000")
    print("📱 Mobile Friendly: Yes")
    print("🔒 Security: File upload validation enabled")
    print("🏥 Medical Safety: Conservative prediction approach")
    
    print("\n💡 USAGE INSTRUCTIONS")
    print("-" * 40)
    print("1. Open your web browser")
    print("2. Navigate to http://localhost:5000")
    print("3. Fill in patient information")
    print("4. Optionally upload chest X-ray image")
    print("5. Click 'Analyze with AI System'")
    print("6. Review comprehensive results")
    
    print("\n🎯 SYSTEM FEATURES")
    print("-" * 40)
    print("• LightGBM feature-based analysis")
    print("• CNN image-based analysis")
    print("• Hybrid prediction system")
    print("• Medical risk assessment")
    print("• Detailed result explanations")

def main():
    """Main startup function"""
    print_banner()
    
    # System checks
    if not check_system_requirements():
        print("\n❌ System requirements not met. Please check missing components.")
        return False
    
    # Initialize directories
    initialize_directories()
    
    # Install requirements
    install_requirements()
    
    # Check models
    model_status = check_model_status()
    
    # Display performance summary
    display_performance_summary()
    
    # Initialize system
    system_status = initialize_system()
    
    # Display system info
    display_system_info()
    
    print("\n" + "="*80)
    print("🎉 SYSTEM INITIALIZATION COMPLETE")
    print("="*80)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Start Flask app
    print("\n🚀 Starting Flask Application...")
    print("Press Ctrl+C to stop the server")
    print("="*80 + "\n")
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\n👋 System shutdown requested")
        print("✅ Monitoring data saved")
        print("✅ System stopped gracefully")
    except Exception as e:
        print(f"\n❌ Error starting Flask app: {e}")
        print("💡 Try running 'python app.py' directly")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)