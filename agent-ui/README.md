# Chat with DB - Frontend

This is the React frontend for the Chat with DB application, which provides a natural language interface for querying databases.

## 🚀 Live Demo

The frontend is deployed and accessible at: **https://arjunseeramsetty.github.io/Chat-With-DB**

## 🛠️ Local Development

### Prerequisites
- Node.js (version 18 or higher)
- npm or yarn

### Installation

1. Navigate to the frontend directory:
```bash
cd agent-ui
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The application will open at `http://localhost:3000`

### Building for Production

```bash
npm run build
```

This creates a `build` folder with the production-ready files.

## 🎨 Features

### Enhanced Chart Visualization
- **Dual-Axis Charts**: Automatically detects and renders dual-axis charts for data with different scales
- **AI-Powered Recommendations**: Backend AI analyzes data and suggests optimal chart configurations
- **Multiple Chart Types**: Line, Bar, Pie, Multi-Line, and Dual-Axis charts
- **Interactive Chart Builder**: Customize chart configurations with a visual interface

### Key Components
- **QueryInput**: Natural language input with speech recognition
- **DataVisualization**: Advanced chart rendering with dual-axis support
- **ClarificationInterface**: Interactive clarification for ambiguous queries
- **ChartBuilder**: Visual chart configuration tool

### Chart Types Supported
- `dualAxisBarLine`: Bar chart on primary axis, line on secondary axis
- `dualAxisLine`: Dual-axis line chart for time series
- `multiLine`: Multi-line chart for grouped data
- `line`: Standard line chart
- `bar`: Standard bar chart
- `pie`: Pie chart for categorical data

## 🔧 Configuration

### Backend Connection
The frontend connects to the backend API. Make sure the backend is running and accessible.

### Environment Variables
Create a `.env` file in the `agent-ui` directory:
```
REACT_APP_API_URL=http://localhost:8000
```

## 🚀 Deployment

The frontend is automatically deployed to GitHub Pages when changes are pushed to the main branch. The deployment is handled by GitHub Actions.

### Manual Deployment
If you need to deploy manually:

1. Build the application:
```bash
npm run build
```

2. Deploy to GitHub Pages:
```bash
npm run deploy
```

## 📁 Project Structure

```
agent-ui/
├── public/                 # Static files
├── src/
│   ├── components/         # React components
│   │   ├── DataVisualization.js  # Chart rendering
│   │   ├── QueryInput.js         # Query input interface
│   │   ├── ChartBuilder.js       # Chart configuration
│   │   └── ...
│   ├── hooks/             # Custom React hooks
│   ├── reducers/          # State management
│   └── App.js             # Main application component
├── package.json           # Dependencies and scripts
└── README.md             # This file
```

## 🎯 Key Features

### Dual-Axis Chart Detection
The application intelligently detects when data has both absolute values and growth/percentage data, automatically suggesting dual-axis charts for better visualization.

### AI-Powered Chart Recommendations
The backend AI analyzes your data and query to recommend the most appropriate chart type and configuration.

### Interactive Chart Builder
Users can customize chart configurations through a visual interface, including:
- X-axis and Y-axis selection
- Secondary axis configuration
- Chart type selection
- Grouping options

## 🔗 Related Links

- **Backend Repository**: [Chat-With-DB Backend](https://github.com/ArjunSeeramsetty/Chat-With-DB)
- **Live Demo**: [https://arjunseeramsetty.github.io/Chat-With-DB](https://arjunseeramsetty.github.io/Chat-With-DB)
- **Documentation**: See the main repository README for comprehensive documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
