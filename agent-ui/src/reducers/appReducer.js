// Initial state for the app
export const initialState = {
  question: "",
  loading: false,
  table: { headers: [], rows: [], chartData: [] },
  summary: "",
  sql: "",
  error: "",
  llm: "openai", // Default to OpenAI for faster responses
  selectedEndpoint: "ask-fixed", // API endpoint selector: ask (enhanced default), ask-enhanced, ask-fixed (traditional), ask-agentic
  plot: null,
  viewMode: "chart",
  suggestions: [],
  history: [],
  selectedChartType: "auto",
  clarificationNeeded: false,
  clarificationQuestion: "",
  clarificationAnswers: {}, // Store clarification answers
  clarificationAnswer: "", // Current clarification response input
  abortController: null, // For cancelling requests
  // Chart builder state
  showChartBuilder: false,
  chartConfig: {
    xAxis: "",
    yAxis: [],
    yAxisSecondary: [],
    secondaryAxis: "", // Single field for secondary axis
    groupBy: "",
    chartType: "auto"
  },
  availableColumns: [],
  // Backend status
  backendStatus: "unknown", // "unknown", "healthy", "unhealthy", "checking"
};

// App reducer function to handle all state transitions
export function appReducer(state, action) {
  switch (action.type) {
    case 'START_QUERY':
      return { 
        ...state, 
        loading: true, 
        error: "",
        table: { headers: [], rows: [], chartData: [] },
        summary: "",
        sql: "",
        plot: null,
        viewMode: "chart",
        suggestions: [],
        showChartBuilder: false,
        abortController: action.payload?.abortController || null
      };
      
    case 'SET_FIELD':
      return { ...state, [action.field]: action.payload };
      
    case 'QUERY_SUCCESS':
      console.log("=== QUERY_SUCCESS ===");
      console.log("Plot data:", action.payload.plot);
      console.log("Plot options:", action.payload.plot?.options);
      
      const updatedState = {
        ...state,
        loading: false,
        error: "",
        question: action.payload.currentQuestion || state.question,
        sql: action.payload.sql || "",
        summary: action.payload.summary || "",
        plot: action.payload.plot || null,
        table: action.payload.table || null,
        confidence: action.payload.confidence || 0.0,
        // Update chartConfig from backend visualization
        chartConfig: action.payload.plot?.options ? {
          ...state.chartConfig,
          chartType: action.payload.plot.chartType || state.chartConfig.chartType,
          xAxis: action.payload.plot.options.xAxis || state.chartConfig.xAxis,
          yAxis: action.payload.plot.options.yAxis || state.chartConfig.yAxis,
          yAxisSecondary: action.payload.plot.options.yAxisSecondary || state.chartConfig.yAxisSecondary,
          secondaryAxis: action.payload.plot.options.secondaryAxis || state.chartConfig.secondaryAxis,
          groupBy: action.payload.plot.options.groupBy || state.chartConfig.groupBy,
          title: action.payload.plot.options.title || state.chartConfig.title,
          description: action.payload.plot.options.description || state.chartConfig.description,
          dataType: action.payload.plot.options.dataType || state.chartConfig.dataType
        } : state.chartConfig,
        // Only add to history if we have a successful response and it's not a clarification
        history: (action.payload.sql && !action.payload.clarification_needed) ? 
          [...state.history, { 
            question: action.payload.currentQuestion || state.question, 
            sql: action.payload.sql,
            summary: action.payload.summary 
          }] : state.history,
        clarificationNeeded: action.payload.clarification_needed || false,
        clarificationQuestion: action.payload.clarification_question || "",
        clarificationAttemptCount: action.payload.clarification_attempt_count || 0,
        // Clear clarification answers if this is a successful response (not a clarification)
        clarificationAnswers: action.payload.clarification_needed ? state.clarificationAnswers : {},
        clarificationAnswer: action.payload.clarification_needed ? state.clarificationAnswer : "",
      };
      
      console.log("Updated clarificationAnswers:", updatedState.clarificationAnswers);
      return updatedState;
      
    case 'QUERY_ERROR':
      return { ...state, loading: false, error: action.payload };
      
    case 'CLEAR_ERROR':
      return { ...state, error: "" };
      
    case 'SET_CLARIFICATION_ANSWER':
      console.log("=== SET_CLARIFICATION_ANSWER ===");
      console.log("Previous clarificationAnswers:", state.clarificationAnswers);
      console.log("New answer:", action.payload);
      const newState = { 
        ...state, 
        clarificationAnswers: { 
          ...state.clarificationAnswers, 
          [action.payload.question]: action.payload.answer 
        } 
      };
      console.log("Updated clarificationAnswers:", newState.clarificationAnswers);
      return newState;
      
    case 'CLEAR_CLARIFICATION':
      return { 
        ...state, 
        clarificationNeeded: false,
        clarificationQuestion: "",
        clarificationAnswers: {},
        clarificationAnswer: ""
      };
      
    case 'CANCEL_REQUEST':
      return { 
        ...state, 
        loading: false,
        error: "Request was cancelled",
        abortController: null
      };
      
    case 'CLEAR_CONVERSATION':
      return { 
        ...state, 
        history: [], 
        clarificationAnswers: {},
        clarificationAnswer: "",
        showChartBuilder: false
      };
      
    case 'TOGGLE_CHART_BUILDER':
      return { ...state, showChartBuilder: !state.showChartBuilder };
      
    case 'UPDATE_CHART_CONFIG':
      return { 
        ...state, 
        chartConfig: { ...state.chartConfig, ...action.payload }
      };
      
    case 'APPLY_CHART_CONFIG':
      return {
        ...state,
        plot: {
          chartType: state.chartConfig.chartType,
          options: {
            title: `Custom Chart: ${state.question}`,
            xAxis: state.chartConfig.xAxis,
            yAxis: state.chartConfig.yAxis,
            yAxisSecondary: state.chartConfig.yAxisSecondary,
            secondaryAxis: state.chartConfig.secondaryAxis,
            groupBy: state.chartConfig.groupBy,
            description: "Custom chart configuration",
            recommendedCharts: [state.chartConfig.chartType, "bar", "line"],
            dataType: "custom"
          }
        }
      };
      
    default:
      return state;
  }
} 