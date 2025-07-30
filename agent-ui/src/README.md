# Frontend Modular Architecture

This document describes the modular structure of the Chat with DB frontend application.

## 📁 Directory Structure

```
src/
├── components/           # Reusable UI components
│   ├── ClarificationInterface.js
│   ├── QueryInput.js
│   └── DataVisualization.js
├── hooks/               # Custom React hooks
│   ├── useClarification.js
│   └── useQueryService.js
├── reducers/            # State management
│   └── appReducer.js
├── App.js               # Original monolithic app
├── App.modular.js       # New modular app
└── README.md           # This file
```

## 🧩 Components

### ClarificationInterface.js
Handles the interactive clarification system when queries need more context.

**Props:**
- `clarificationQuestion`: The question from the backend
- `clarificationAnswer`: Current user input
- `loading`: Loading state
- `onClarificationResponse`: Submit handler
- `onClarificationChange`: Input change handler
- `onCancel`: Cancel handler

### QueryInput.js
Manages user input including text and voice recognition.

**Props:**
- `question`: Current question text
- `onQuestionChange`: Text change handler
- `onAsk`: Submit handler
- `onCancel`: Cancel handler
- `loading`: Loading state
- `listening`: Voice recognition state
- `onStartListening`: Start voice input
- `onStopListening`: Stop voice input
- `browserSupportsSpeechRecognition`: Browser compatibility
- `backendStatus`: Backend health status

### DataVisualization.js
Renders charts and tables based on query results.

**Props:**
- `plot`: Chart configuration
- `table`: Data table
- `viewMode`: Current view mode (chart/table)
- `onViewModeChange`: View mode change handler
- `selectedChartType`: Current chart type
- `onChartTypeChange`: Chart type change handler
- `showChartBuilder`: Chart builder visibility
- `onToggleChartBuilder`: Toggle chart builder
- `onUpdateChartConfig`: Update chart configuration
- `chartConfig`: Current chart configuration
- `availableColumns`: Available data columns

## 🪝 Custom Hooks

### useClarification.js
Manages clarification state and logic.

**Returns:**
- `handleClarificationResponse`: Submit clarification response
- `handleClarificationChange`: Update clarification input
- `handleClarificationCancel`: Cancel clarification
- `isClarificationNeeded`: Check if clarification is needed

### useQueryService.js
Handles API calls and query processing.

**Returns:**
- `handleAsk`: Submit query to backend
- `handleCancel`: Cancel ongoing request
- `checkBackendHealth`: Check backend status

## 🔄 State Management

### appReducer.js
Centralized state management using useReducer.

**Actions:**
- `START_QUERY`: Begin query processing
- `QUERY_SUCCESS`: Handle successful response
- `QUERY_ERROR`: Handle error response
- `SET_CLARIFICATION_ANSWER`: Store clarification response
- `CLEAR_CLARIFICATION`: Clear clarification state
- `CANCEL_REQUEST`: Cancel ongoing request
- `CLEAR_CONVERSATION`: Clear conversation history
- `TOGGLE_CHART_BUILDER`: Toggle chart builder
- `UPDATE_CHART_CONFIG`: Update chart configuration

## 🚀 Usage

### Using the Modular App
Replace the import in `index.js`:

```javascript
// Instead of:
import App from './App';

// Use:
import App from './App.modular';
```

### Adding New Components
1. Create component in `components/` directory
2. Export as default
3. Import and use in `App.modular.js`

### Adding New Hooks
1. Create hook in `hooks/` directory
2. Export as default
3. Import and use in `App.modular.js`

### Adding New Reducer Actions
1. Add action type and case in `reducers/appReducer.js`
2. Dispatch action from components or hooks

## 🔧 Benefits of Modular Structure

1. **Separation of Concerns**: Each component has a single responsibility
2. **Reusability**: Components can be reused across different parts of the app
3. **Testability**: Individual components and hooks can be tested in isolation
4. **Maintainability**: Easier to find and fix issues
5. **Scalability**: Easy to add new features without affecting existing code
6. **Clarification System**: Dedicated component for handling user clarifications

## 🎯 Clarification System

The clarification system is now fully modular and interactive:

1. **Backend Integration**: Sends clarification questions to frontend
2. **Interactive Interface**: Users can type responses
3. **Context Preservation**: Maintains conversation context
4. **Error Handling**: Proper error states and validation
5. **User Experience**: Clear visual feedback and instructions

## 📝 Migration Guide

To migrate from the monolithic `App.js` to the modular `App.modular.js`:

1. **Backup**: Keep the original `App.js` as backup
2. **Test**: Ensure all functionality works in modular version
3. **Update**: Replace import in `index.js`
4. **Cleanup**: Remove unused code from original `App.js`

## 🔮 Future Enhancements

- **Component Testing**: Add unit tests for each component
- **TypeScript**: Convert to TypeScript for better type safety
- **State Management**: Consider Redux for complex state
- **Performance**: Add React.memo for performance optimization
- **Accessibility**: Improve accessibility features 