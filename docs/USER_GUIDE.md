# 🏎️ F1 Tire Thermal Platform

## User Guide & Operations Manual

**Version**: 2.0  
**Last Updated**: September 2025  
**Target Audience**: F1 Engineers, Race Strategists, Data Analysts, Operations Teams

---

## 📋 **Table of Contents**

1. [System Overview](#system-overview)
2. [Getting Started](#getting-started)
3. [Dashboard Navigation](#dashboard-navigation)
4. [Live Telemetry Monitoring](#live-telemetry-monitoring)
5. [Advanced Analytics](#advanced-analytics)
6. [Multi-Driver Management](#multi-driver-management)
7. [Environmental Intelligence](#environmental-intelligence)
8. [Strategy Optimization](#strategy-optimization)
9. [Data Export & Reporting](#data-export--reporting)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)

---

## 🎯 **System Overview**

The F1 Tire Thermal Platform is a professional-grade platform that provides comprehensive tire thermal modeling, strategy optimization, and real-time analytics for Formula 1 racing applications.

### **Key Capabilities**

- **Real-Time Thermal Monitoring**: 3-node thermal modeling with EKF
- **Advanced Analytics**: Big data processing and predictive insights
- **Machine Learning**: Strategy optimization and predictive modeling
- **Multi-Driver Support**: Driver profiling and personalized recommendations
- **Environmental Intelligence**: Weather integration and track evolution
- **Professional Dashboard**: F1-themed operations center interface

---

## 🚀 **Getting Started**

### **System Requirements**

- **Operating System**: Windows 10+, macOS 10.15+, Linux Ubuntu 18.04+
- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 2GB available space
- **Network**: Internet connection for weather data

### **Installation**

```bash
# Clone the repository
git clone https://github.com/ac12644/f1-tire-temp-prototype.git
cd f1-tire-temp-prototype

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run src/app_streamlit.py
```

### **First Launch**

1. **Open Browser**: Navigate to `http://localhost:8501`
2. **Initialize Session**: Click "Initialize Session" if prompted
3. **Configure Settings**: Set up your preferences in the sidebar
4. **Start Monitoring**: Click "Run" to begin live telemetry

---

## 🎛️ **Dashboard Navigation**

### **Main Interface Layout**

```
┌─────────────────────────────────────────────────────────────┐
│  🏎️ F1 TIRE MANAGEMENT SYSTEM                              │
│  Professional Grade Tire Temperature & Strategy Platform   │
├─────────────────────────────────────────────────────────────┤
│ Sidebar Controls    │ Main Dashboard Area                  │
│ ┌─────────────────┐ │ ┌─────────────────────────────────────┐ │
│ │ View Selection  │ │ │ Tab Navigation                    │ │
│ │ Driver Profile  │ │ │ ┌─────┬─────┬─────┬─────┬─────┐   │ │
│ │ Weather Controls│ │ │ │Live│Wear │Weather│Driver│Adv│ │ │
│ │ Modelpack       │ │ │ └─────┴─────┴─────┴─────┴─────┘   │ │
│ │ Export Options  │ │ │                                   │ │
│ └─────────────────┘ │ │ Content Area                      │ │
│                     │ │                                   │ │
│                     │ │                                   │ │
│                     │ └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### **Sidebar Controls**

#### **View Selection**

- **Live**: Real-time telemetry monitoring
- **What-If**: Forward simulation and scenario testing
- **Session**: Event log and session management
- **Modelpack**: Configuration management
- **Export**: Data export and reporting

#### **Driver Profile**

- **Driver Selection**: Choose active driver
- **Profile Display**: Driver characteristics and preferences
- **Personalized Settings**: Driver-specific parameters

#### **Weather Controls**

- **Rain Probability**: 0-100% slider
- **Wind Speed**: 0-50 km/h slider
- **Humidity**: 0-100% slider
- **Session Type**: FP1, FP2, FP3, Qualifying, Race

---

## 📊 **Live Telemetry Monitoring**

### **Temperature Monitoring**

#### **Real-Time Charts**

- **Tread Temperature**: Outer rubber layer temperature
- **Carcass Temperature**: Tire structure temperature (estimated via EKF)
- **Rim Temperature**: Metal wheel temperature

#### **Temperature Bands**

- **Soft Compound**: 95-110°C (optimal range)
- **Medium Compound**: 90-106°C (optimal range)
- **Hard Compound**: 88-104°C (optimal range)

#### **Status Indicators**

- **🟢 Optimal**: Temperature within optimal range
- **🟡 Warning**: Temperature approaching limits
- **🔴 Critical**: Temperature outside safe range

### **Wear Monitoring**

#### **Wear Levels**

- **Front Left (FL)**: Individual corner wear tracking
- **Front Right (FR)**: Individual corner wear tracking
- **Rear Left (RL)**: Individual corner wear tracking
- **Rear Right (RR)**: Individual corner wear tracking

#### **Wear Effects**

- **Grip Degradation**: Real-time grip reduction calculation
- **Stiffness Reduction**: Tire stiffness impact
- **Pit Window Prediction**: Optimal pit stop timing

### **Recommendations Engine**

#### **Thermal Recommendations**

- **Temperature Management**: Adjust driving style
- **Cooling Strategies**: Brake bias adjustments
- **Heating Strategies**: Slip angle management

#### **Wear-Based Recommendations**

- **Wear Management**: Conservative vs aggressive driving
- **Pit Stop Timing**: Optimal pit window predictions
- **Compound Strategy**: Tire compound recommendations

---

## 📈 **Advanced Analytics**

### **Big Data Analytics**

#### **Performance Report Generation**

1. **Click "Generate Analytics Report"**
2. **Wait for Processing**: System analyzes historical data
3. **Review Results**: Comprehensive performance analysis
4. **Export Data**: Download CSV for external analysis

#### **Analytics Metrics**

- **Data Points Stored**: Total telemetry records
- **Performance Records**: Analysis data points
- **Weather Records**: Environmental data
- **Driver Records**: Driver performance data

### **Predictive Analytics**

#### **Lap Time Prediction**

1. **Click "Generate Predictive Analysis"**
2. **System Processes**: Current thermal state, wear levels, weather
3. **Prediction Results**: Lap time estimates with confidence intervals
4. **Factors Analysis**: Contributing factors to prediction

#### **Degradation Prediction**

- **Wear Forecasting**: Future wear level predictions
- **Grip Projection**: Expected grip degradation
- **Pit Window Analysis**: Optimal pit stop timing

### **Strategy Optimization**

#### **Race Strategy Optimization**

1. **Click "Optimize Race Strategy"**
2. **Genetic Algorithm**: ML-based optimization process
3. **Strategy Results**: Optimal pit windows and tire choices
4. **Confidence Score**: Strategy reliability assessment

#### **Strategy Parameters**

- **Pit Windows**: Optimal pit stop timing
- **Tire Strategy**: Compound selection and timing
- **Driving Style**: Aggressive vs conservative approach
- **Fuel Strategy**: Fuel load optimization

### **Data-Driven Insights**

#### **Insight Generation**

1. **Click "Generate Insights"**
2. **Pattern Recognition**: Automated insight generation
3. **Priority Levels**: High, Medium, Low priority insights
4. **Categories**: Optimization, Anomaly, Pattern insights

#### **Insight Types**

- **Thermal Efficiency**: Temperature management optimization
- **Wear Optimization**: Tire usage optimization
- **Lap Time Consistency**: Performance consistency analysis
- **Anomaly Detection**: Unusual patterns and events

---

## 👥 **Multi-Driver Management**

### **Driver Selection**

#### **Available Drivers**

- **Lewis Hamilton**: Aggressive style, high thermal aggression
- **Max Verstappen**: Adaptive style, balanced parameters
- **Charles Leclerc**: Conservative style, high tire awareness
- **Lando Norris**: Rookie profile, learning parameters

#### **Driver Profile Display**

- **Style**: Aggressive, Conservative, Adaptive
- **Experience**: Rookie, Intermediate, Veteran
- **Thermal Aggression**: 0.0-1.0 scale
- **Tire Awareness**: 0.0-1.0 scale

### **Personalized Recommendations**

#### **Driver-Specific Temperature Bands**

- **Aggressive Drivers**: Wider optimal ranges
- **Conservative Drivers**: Narrower optimal ranges
- **Adaptive Drivers**: Dynamic range adjustment

#### **Personalized Strategies**

- **Thermal Management**: Driver-specific cooling/heating strategies
- **Wear Management**: Personalized wear tolerance levels
- **Pit Strategy**: Driver-specific pit window preferences

### **Driver Comparison**

#### **Performance Metrics**

- **Thermal Efficiency**: Temperature management effectiveness
- **Wear Management**: Tire usage optimization
- **Consistency**: Performance stability metrics
- **Adaptability**: Weather condition adaptation

#### **Rankings**

- **Overall Performance**: Combined metric ranking
- **Thermal Management**: Temperature control ranking
- **Wear Optimization**: Tire usage ranking
- **Strategy Execution**: Pit stop timing ranking

---

## 🌤️ **Environmental Intelligence**

### **Weather Integration**

#### **Weather Controls**

- **Rain Probability**: Set expected rain probability (0-100%)
- **Wind Speed**: Configure wind conditions (0-50 km/h)
- **Humidity**: Set humidity levels (0-100%)

#### **Weather Effects**

- **Thermal Impact**: Weather effects on tire temperatures
- **Grip Modification**: Weather impact on grip levels
- **Cooling Enhancement**: Wind effects on cooling

### **Track Evolution**

#### **Session Progression**

- **Rubbering In**: Track grip improvement over session
- **Temperature Evolution**: Track temperature changes
- **Surface Conditions**: Dry, wet, mixed conditions

#### **Environmental Sensors**

- **Ambient Temperature**: Air temperature monitoring
- **Track Temperature**: Surface temperature tracking
- **Atmospheric Pressure**: Pressure monitoring
- **Wind Direction**: Wind direction tracking

### **Session Management**

#### **Session Types**

- **FP1 (Free Practice 1)**: Early session, track evolution
- **FP2 (Free Practice 2)**: Mid-session, stable conditions
- **FP3 (Free Practice 3)**: Late session, race simulation
- **Qualifying**: High-intensity, optimal conditions
- **Race**: Full race conditions, strategy execution

---

## 🎯 **Strategy Optimization**

### **Race Simulation**

#### **Simulation Engine**

- **Scenario Modeling**: Multiple race scenarios
- **Strategy Testing**: Pit stop strategy validation
- **Performance Prediction**: Race outcome forecasting
- **Risk Assessment**: Strategy risk evaluation

#### **Simulation Parameters**

- **Race Duration**: Total race laps
- **Weather Conditions**: Expected weather changes
- **Tire Strategy**: Starting compound and pit strategy
- **Fuel Strategy**: Fuel load and consumption

### **Genetic Algorithm Optimization**

#### **Optimization Process**

1. **Population Initialization**: Generate initial strategy population
2. **Fitness Evaluation**: Evaluate each strategy's performance
3. **Selection**: Select best-performing strategies
4. **Crossover**: Combine strategies to create offspring
5. **Mutation**: Introduce random variations
6. **Convergence**: Repeat until optimal solution found

#### **Strategy Chromosomes**

- **Pit Windows**: Optimal pit stop timing
- **Tire Pressure**: Optimal tire pressure settings
- **Driving Style**: Aggressive vs conservative approach
- **Fuel Strategy**: Fuel load optimization

---

## 📊 **Data Export & Reporting**

### **CSV Export**

#### **Race Summary Export**

- **Report Type**: Race summary data
- **Generated By**: F1 Tire Management System
- **Metrics**: Current lap, compound, temperatures, wear levels
- **Timestamp**: Generation timestamp

#### **Performance Analysis Export**

- **Report Type**: Performance analysis data
- **Metrics**: Speed, lap time, thermal data, wear levels
- **Units**: Proper units for each metric
- **Timestamp**: Generation timestamp

#### **Tire Temperature Data Export**

- **Historical Data**: Complete temperature history
- **Corner Data**: Individual corner tracking
- **Time Series**: Timestamped data points
- **Export Format**: CSV with proper headers

### **Professional Reports**

#### **Report Generation**

- **Automated Reports**: Scheduled report generation
- **Custom Reports**: User-defined report templates
- **Multi-Format**: CSV, HTML, PDF export options
- **Real-Time**: Live report generation

#### **Report Types**

- **Performance Analysis**: Comprehensive performance reports
- **Predictive Insights**: ML-based predictions and insights
- **Strategy Optimization**: Optimal strategy recommendations
- **Anomaly Detection**: Unusual pattern identification

---

## 🔧 **Troubleshooting**

### **Common Issues**

#### **Application Won't Start**

- **Check Python Version**: Ensure Python 3.8+ is installed
- **Install Dependencies**: Run `pip install -r requirements.txt`
- **Check Port**: Ensure port 8501 is available
- **Restart Application**: Close and restart Streamlit

#### **Temperature Readings Unrealistic**

- **Check Modelpack**: Verify correct modelpack is loaded
- **Reset Simulation**: Click "Reset" to restart simulation
- **Check Parameters**: Verify thermal parameters are correct
- **Update Modelpack**: Try different modelpack configuration

#### **Recommendations Not Appearing**

- **Check Temperature Bands**: Ensure temperatures are outside optimal range
- **Verify Wear Levels**: Check if wear levels exceed thresholds
- **Update Weather**: Ensure weather model is active
- **Check Driver Profile**: Verify driver profile is selected

#### **Export Issues**

- **Check Browser**: Ensure browser allows downloads
- **Clear Cache**: Clear browser cache and try again
- **Check Permissions**: Verify file system permissions
- **Try Different Format**: Test different export formats

### **Performance Issues**

#### **Slow Response Times**

- **Reduce Data Points**: Limit historical data retention
- **Close Other Applications**: Free up system resources
- **Check Memory Usage**: Monitor system memory usage
- **Update Hardware**: Consider hardware upgrades

#### **High CPU Usage**

- **Reduce Update Frequency**: Increase time step intervals
- **Disable Advanced Features**: Turn off ML features temporarily
- **Check Background Processes**: Close unnecessary applications
- **Optimize Settings**: Adjust performance settings

---

## 🏆 **Best Practices**

### **Operational Excellence**

#### **Daily Operations**

1. **System Startup**: Initialize session and verify all components
2. **Weather Check**: Update weather conditions and forecasts
3. **Driver Selection**: Choose appropriate driver profile
4. **Modelpack Verification**: Ensure correct track configuration
5. **Monitoring Setup**: Configure alerts and thresholds

#### **Session Management**

1. **Pre-Session**: Configure weather and track conditions
2. **During Session**: Monitor real-time data and recommendations
3. **Post-Session**: Export data and generate reports
4. **Analysis**: Review performance and identify improvements

### **Data Management**

#### **Data Quality**

- **Regular Validation**: Verify data accuracy and completeness
- **Anomaly Detection**: Monitor for unusual patterns
- **Backup Procedures**: Regular data backup and recovery
- **Retention Policies**: Manage data retention and cleanup

#### **Analysis Workflow**

1. **Data Collection**: Gather comprehensive telemetry data
2. **Data Processing**: Clean and validate data
3. **Analysis**: Apply analytical methods and ML models
4. **Insights**: Generate actionable insights and recommendations
5. **Implementation**: Apply insights to improve performance

### **Team Collaboration**

#### **Role-Based Access**

- **Engineers**: Full system access and configuration
- **Strategists**: Analytics and strategy optimization access
- **Drivers**: Personalized recommendations and feedback
- **Management**: High-level reporting and insights

#### **Communication Protocols**

- **Real-Time Updates**: Live data sharing and notifications
- **Decision Making**: Collaborative decision processes
- **Documentation**: Comprehensive operation documentation
- **Training**: Regular training and skill development

### **Continuous Improvement**

#### **Performance Monitoring**

- **KPI Tracking**: Monitor key performance indicators
- **Benchmarking**: Compare against industry standards
- **Trend Analysis**: Identify performance trends and patterns
- **Optimization**: Continuous system optimization

#### **System Updates**

- **Regular Updates**: Keep system components current
- **Feature Enhancement**: Add new capabilities and features
- **Bug Fixes**: Address issues and improve reliability
- **User Feedback**: Incorporate user feedback and suggestions

---

## 📞 **Support & Resources**

### **Technical Support**

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Comprehensive technical documentation
- **Code Examples**: Sample implementations and use cases
- **Community Forum**: Developer community support

### **Training Resources**

- **User Manual**: This comprehensive guide
- **Video Tutorials**: Step-by-step video instructions
- **Webinars**: Live training sessions
- **Certification**: Professional certification programs

### **Professional Services**

- **Custom Development**: Tailored solutions for teams
- **Training Programs**: Professional training courses
- **Consulting Services**: Expert consultation and support
- **Integration Support**: System integration assistance

---

**User Guide Version**: 2.0  
**Last Updated**: September 2025  
**Next Review**: September 2026

_This user guide provides comprehensive instructions for operating the F1 Tire Temperature Management System effectively and efficiently._
