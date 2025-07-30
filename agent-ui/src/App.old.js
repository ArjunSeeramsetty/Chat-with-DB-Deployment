import React, { useState, useReducer, useEffect, useCallback } from "react";
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import {
  Container,
  TextField,
  Button,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  Alert,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Tabs,
  Tab,
  Box,
  Chip,
  Stack,
  IconButton,
  Tooltip,
  Grid,
  Card,
  CardContent,
  Divider,
} from "@mui/material";
import MicIcon from '@mui/icons-material/Mic';
import MicOffIcon from '@mui/icons-material/MicOff';
import DragIndicatorIcon from '@mui/icons-material/DragIndicator';
import ReactMarkdown from "react-markdown";
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer, ComposedChart
} from 'recharts';
import { DataGrid } from '@mui/x-data-grid';
import { ContentCopy, CheckCircle } from '@mui/icons-material';
import useQueryService from './hooks/useQueryService';

// Initial state for the app
const initialState = {
  question: "",
  loading: false,
  table: { headers: [], rows: [], chartData: [] },
  summary: "",
  sql: "",
  error: "",
  llm: "openai", // Default to OpenAI for faster responses
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
    yAxisSecondary: "",
    groupBy: "",
    chartType: "auto"
  },
  availableColumns: [],
  // Backend status
  backendStatus: "unknown", // "unknown", "healthy", "unhealthy", "checking"
};

// Reducer function to handle all state transitions
function appReducer(state, action) {
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
      return {
        ...state,
        loading: false,
        sql: action.payload.sql || "",
        summary: action.payload.summary || "",
        plot: action.payload.plot || null,
        suggestions: action.payload.suggestions || [],
        table: action.payload.table || { headers: [], rows: [], chartData: [] },
        viewMode: action.payload.plot ? 'chart' : 'table',
        clarification_needed: action.payload.clarification_needed || false,
        clarification_question: action.payload.clarification_question || "",
        // Set up chart builder with available columns
        availableColumns: action.payload.table?.headers || [],
        chartConfig: action.payload.plot?.options?.dataType === "growth_time_series" ? {
          xAxis: action.payload.plot?.options?.xAxis || "GrowthMonth",
          yAxis: action.payload.plot?.options?.yAxis || ["GrowthPercent"],
          groupBy: action.payload.plot?.options?.groupBy || "SourceName",
          chartType: "multiLine"
        } : {
          xAxis: action.payload.plot?.options?.xAxis || "",
          yAxis: action.payload.plot?.options?.yAxis || [],
          groupBy: action.payload.plot?.options?.groupBy || "",
          chartType: action.payload.plot?.chartType || "auto"
        },
        // Only add to history if we have a successful response and it's not a clarification
        history: (action.payload.sql && !action.payload.clarification_needed) ? 
          [...state.history, { 
            question: action.payload.currentQuestion || state.question, 
            sql: action.payload.sql,
            summary: action.payload.summary 
          }] : state.history,
        clarificationNeeded: action.payload.clarification_needed || false,
        clarificationQuestion: action.payload.clarification_question || "",
      };
    case 'QUERY_ERROR':
      return { ...state, loading: false, error: action.payload };
    case 'CLEAR_ERROR':
      return { ...state, error: "" };
    case 'SET_CLARIFICATION_ANSWER':
      return { 
        ...state, 
        clarificationAnswers: { 
          ...state.clarificationAnswers, 
          [action.payload.question]: action.payload.answer 
        } 
      };
    case 'CLEAR_CLARIFICATION':
      return { 
        ...state, 
        clarificationNeeded: false,
        clarificationQuestion: "",
        clarificationAnswers: {}
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
            groupBy: state.chartConfig.groupBy,
            description: "Custom chart configuration",
            recommendedCharts: [state.chartConfig.chartType, "bar", "line"],
            dataType: "custom_configuration"
          }
        },
        showChartBuilder: false
      };
    default:
      return state;
  }
}

// Chart Builder Component
function ChartBuilder({ state, dispatch }) {
  const handleDragStart = (e, column) => {
    e.dataTransfer.setData('text/plain', column);
  };

  const handleDrop = (e, target) => {
    e.preventDefault();
    const column = e.dataTransfer.getData('text/plain');
    
    if (target === 'xAxis') {
      dispatch({ 
        type: 'UPDATE_CHART_CONFIG', 
        payload: { xAxis: column } 
      });
    } else if (target === 'yAxis') {
      const currentYAxis = state.chartConfig.yAxis;
      if (!currentYAxis.includes(column)) {
        dispatch({ 
          type: 'UPDATE_CHART_CONFIG', 
          payload: { yAxis: [...currentYAxis, column] } 
        });
      }
    } else if (target === 'yAxisSecondary') {
      dispatch({ 
        type: 'UPDATE_CHART_CONFIG', 
        payload: { yAxisSecondary: column } 
      });
    } else if (target === 'groupBy') {
      dispatch({ 
        type: 'UPDATE_CHART_CONFIG', 
        payload: { groupBy: column } 
      });
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const removeFromYAxis = (column) => {
    const newYAxis = state.chartConfig.yAxis.filter(col => col !== column);
    dispatch({ 
      type: 'UPDATE_CHART_CONFIG', 
      payload: { yAxis: newYAxis } 
    });
  };

  const clearArea = (area) => {
    if (area === 'xAxis') {
      dispatch({ type: 'UPDATE_CHART_CONFIG', payload: { xAxis: "" } });
    } else if (area === 'yAxis') {
      dispatch({ type: 'UPDATE_CHART_CONFIG', payload: { yAxis: [] } });
    } else if (area === 'groupBy') {
      dispatch({ type: 'UPDATE_CHART_CONFIG', payload: { groupBy: "" } });
    }
  };

  return (
      <Paper style={{ padding: 16, marginBottom: 16 }}>
        <Typography variant="h6" gutterBottom>
          🎨 Chart Builder - Drag & Drop Interface
        </Typography>
        
        {/* Instructions for Multi-Line Charts */}
        {state.chartConfig.chartType === "multiLine" && (
          <Alert severity="info" sx={{ mb: 2 }}>
            <Typography variant="body2">
              <strong>Multi-Line Chart Setup:</strong> For proper multi-line charts, drag:
              <br />• <strong>Time/Month</strong> column to X-Axis
              <br />• <strong>Value</strong> column to Y-Axis  
              <br />• <strong>Category/Source</strong> column to Group By
            </Typography>
          </Alert>
        )}
      
      <Grid container spacing={2}>
        {/* Available Columns */}
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                📋 Available Columns
              </Typography>
              <Stack spacing={1}>
                {state.availableColumns.map((column) => (
                  <Chip
                    key={column}
                    label={column}
                    draggable
                    onDragStart={(e) => handleDragStart(e, column)}
                    icon={<DragIndicatorIcon />}
                    variant="outlined"
                    style={{ cursor: 'grab' }}
                  />
                ))}
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* Chart Configuration Areas */}
        <Grid item xs={12} md={9}>
          <Grid container spacing={2}>
            {/* X-Axis */}
            <Grid item xs={12} md={4}>
              <Card 
                onDrop={(e) => handleDrop(e, 'xAxis')}
                onDragOver={handleDragOver}
                style={{ 
                  minHeight: 120, 
                  border: '2px dashed #ccc',
                  backgroundColor: state.chartConfig.xAxis ? '#e8f5e8' : '#f5f5f5'
                }}
              >
                <CardContent>
                  <Typography variant="subtitle2" gutterBottom>
                    📊 X-Axis
                  </Typography>
                  {state.chartConfig.xAxis ? (
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <Chip label={state.chartConfig.xAxis} color="primary" />
                      <IconButton size="small" onClick={() => clearArea('xAxis')}>
                        ×
                      </IconButton>
                    </Box>
                  ) : (
                    <Typography variant="body2" color="textSecondary">
                      Drop column here
                    </Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>

            {/* Y-Axis */}
            <Grid item xs={12} md={4}>
              <Card 
                onDrop={(e) => handleDrop(e, 'yAxis')}
                onDragOver={handleDragOver}
                style={{ 
                  minHeight: 120, 
                  border: '2px dashed #ccc',
                  backgroundColor: state.chartConfig.yAxis.length > 0 ? '#e8f5e8' : '#f5f5f5'
                }}
              >
                <CardContent>
                  <Typography variant="subtitle2" gutterBottom>
                    📈 Y-Axis
                  </Typography>
                  {state.chartConfig.yAxis.length > 0 ? (
                    <Stack spacing={1}>
                      {state.chartConfig.yAxis.map((column) => (
                        <Box key={column} sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                          <Chip label={column} color="secondary" size="small" />
                          <IconButton size="small" onClick={() => removeFromYAxis(column)}>
                            ×
                          </IconButton>
                        </Box>
                      ))}
                    </Stack>
                  ) : (
                    <Typography variant="body2" color="textSecondary">
                      Drop columns here
                    </Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>

            {/* Y-Axis Secondary */}
            <Grid item xs={12} md={4}>
              <Card 
                onDrop={(e) => handleDrop(e, 'yAxisSecondary')}
                onDragOver={handleDragOver}
                style={{ 
                  minHeight: 120, 
                  border: '2px dashed #ccc',
                  backgroundColor: state.chartConfig.yAxisSecondary ? '#e8f5e8' : '#f5f5f5'
                }}
              >
                <CardContent>
                  <Typography variant="subtitle2" gutterBottom>
                    📊 Secondary Y-Axis
                  </Typography>
                  {state.chartConfig.yAxisSecondary ? (
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <Chip label={state.chartConfig.yAxisSecondary} color="info" size="small" />
                      <IconButton size="small" onClick={() => dispatch({ type: 'UPDATE_CHART_CONFIG', payload: { yAxisSecondary: "" } })}>
                        ×
                      </IconButton>
                    </Box>
                  ) : (
                    <Typography variant="body2" color="textSecondary">
                      Drop column here
                    </Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>

            {/* Group By */}
            <Grid item xs={12} md={4}>
              <Card 
                onDrop={(e) => handleDrop(e, 'groupBy')}
                onDragOver={handleDragOver}
                style={{ 
                  minHeight: 120, 
                  border: '2px dashed #ccc',
                  backgroundColor: state.chartConfig.groupBy ? '#e8f5e8' : '#f5f5f5'
                }}
              >
                <CardContent>
                  <Typography variant="subtitle2" gutterBottom>
                    🎯 Group By
                  </Typography>
                  {state.chartConfig.groupBy ? (
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <Chip label={state.chartConfig.groupBy} color="success" />
                      <IconButton size="small" onClick={() => clearArea('groupBy')}>
                        ×
                      </IconButton>
                    </Box>
                  ) : (
                    <Typography variant="body2" color="textSecondary">
                      Drop column here
                    </Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Chart Type Selector */}
          <Box sx={{ mt: 2 }}>
            <FormControl fullWidth>
              <InputLabel>Chart Type</InputLabel>
              <Select
                value={state.chartConfig.chartType}
                onChange={(e) => dispatch({ 
                  type: 'UPDATE_CHART_CONFIG', 
                  payload: { chartType: e.target.value } 
                })}
              >
                <MenuItem value="auto">Auto (AI Recommended)</MenuItem>
                <MenuItem value="dualAxisBarLine">Dual-Axis Bar + Line Chart</MenuItem>
                <MenuItem value="dualAxisLine">Dual-Axis Line Chart</MenuItem>
                <MenuItem value="multiLine">Multi-Line Chart</MenuItem>
                <MenuItem value="line">Line Chart</MenuItem>
                <MenuItem value="bar">Bar Chart</MenuItem>
                <MenuItem value="stackedBar">Stacked Bar Chart</MenuItem>
                <MenuItem value="groupedBar">Grouped Bar Chart</MenuItem>
                <MenuItem value="pie">Pie Chart</MenuItem>
              </Select>
            </FormControl>
          </Box>

          {/* Apply Button */}
          <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
            <Button 
              variant="contained" 
              onClick={() => dispatch({ type: 'APPLY_CHART_CONFIG' })}
              disabled={!state.chartConfig.xAxis || state.chartConfig.yAxis.length === 0}
            >
              🎨 Apply Chart Configuration
            </Button>
            <Button 
              variant="outlined" 
              onClick={() => dispatch({ type: 'TOGGLE_CHART_BUILDER' })}
            >
              Cancel
            </Button>
          </Box>
        </Grid>
      </Grid>
    </Paper>
  );
}

function App() {
  const [state, dispatch] = useReducer(appReducer, initialState);
  const [copied, setCopied] = useState(false);
  
  const { handleAsk, loading, error } = useQueryService(state, dispatch);
  
  // Initialize the speech recognition hook
  const {
    transcript,
    listening,
    resetTranscript,
    browserSupportsSpeechRecognition
  } = useSpeechRecognition();

  // Check for browser support
  if (!browserSupportsSpeechRecognition) {
    console.warn("This browser does not support speech recognition.");
  }

  // Effect to capture final transcript when listening stops and auto-submit
  useEffect(() => {
    if (!listening && transcript && transcript.trim().length > 0) {
      dispatch({ type: 'SET_FIELD', field: 'question', payload: transcript });
      // Auto-submit after a short delay to allow user to see the transcribed text
      const submitTimer = setTimeout(() => {
        if (transcript.trim().length > 0) {
          handleAsk();
        }
      }, 1500); // 1.5 second delay to give user time to see the text
      
      return () => clearTimeout(submitTimer);
    }
  }, [listening, transcript, dispatch]);

  // Auto-stop listening after silence timeout
  useEffect(() => {
    let silenceTimer;
    if (listening) {
      silenceTimer = setTimeout(() => {
        SpeechRecognition.stopListening();
      }, 3000); // Stop after 3 seconds of silence
    }
    
    return () => {
      if (silenceTimer) clearTimeout(silenceTimer);
    };
  }, [listening, transcript]);

  // Check backend status on component mount
  useEffect(() => {
    const checkBackendStatus = async () => {
      dispatch({ type: 'SET_FIELD', field: 'backendStatus', payload: 'checking' });
      const isHealthy = await checkBackendHealth();
      dispatch({ 
        type: 'SET_FIELD', 
        field: 'backendStatus', 
        payload: isHealthy ? 'healthy' : 'unhealthy' 
      });
    };
    
    checkBackendStatus();
    
    // Check status every 30 seconds
    const interval = setInterval(checkBackendStatus, 30000);
    
    return () => clearInterval(interval);
  }, []);

  // Helper to determine chart type based on data
  function getChartTypeForData(headers, rows, plotRecommendation) {
    if (!headers || !rows || rows.length === 0) return "bar";
    
    // Use AI recommendation if available
    if (plotRecommendation?.chartType && plotRecommendation.chartType !== "auto") {
      return plotRecommendation.chartType;
    }
    
    const numColumns = headers.length;
    const firstColHeader = headers[0]?.toLowerCase() || "";
    const secondColHeader = headers[1]?.toLowerCase() || "";
    const firstColValues = rows.map(row => row[0]).filter(val => val !== null && val !== undefined);
    
    // Check for growth data (multiple sources with time series)
    const isGrowthData = headers.some(h => h.toLowerCase().includes('growth')) || 
                        headers.some(h => h.toLowerCase().includes('percent'));
    const hasMultipleSources = numColumns >= 3 && headers.some(h => h.toLowerCase().includes('source'));
    const isTimeSeries = ['date', 'month', 'year', 'time', 'day', 'quarter', 'growthmonth'].some(word => 
      firstColHeader.includes(word)
    );
    
    // Check for dual-axis scenarios (different scales)
    const hasTotalValues = headers.some(h => h.toLowerCase().includes('total') || h.toLowerCase().includes('demand') || h.toLowerCase().includes('generation'));
    const hasGrowthValues = headers.some(h => h.toLowerCase().includes('growth') || h.toLowerCase().includes('percent'));
    
    // Dual-axis recommendation for total values + growth percentages
    if (hasTotalValues && hasGrowthValues && numColumns >= 3) {
      return "dualAxisBarLine";
    }
    
    // Check for growth data with total energy (common pattern)
    if (isGrowthData && headers.some(h => h.toLowerCase().includes('energy'))) {
      const energyCol = headers.find(h => h.toLowerCase().includes('energy'));
      const growthCol = headers.find(h => h.toLowerCase().includes('growth') || h.toLowerCase().includes('percent'));
      
      if (energyCol && growthCol && energyCol !== growthCol) {
        return "dualAxisBarLine";
      }
    }
    
    // Check for multiple value columns with different scales
    if (numColumns >= 3) {
      const valueColumns = headers.filter(h => 
        h.toLowerCase().includes('total') || 
        h.toLowerCase().includes('demand') || 
        h.toLowerCase().includes('generation') ||
        h.toLowerCase().includes('energy')
      );
      
      if (valueColumns.length >= 2) {
        // Check if scales are very different
        const scales = [];
        for (const col of valueColumns) {
          const colIndex = headers.indexOf(col);
          const values = rows.map(row => row[colIndex]).filter(val => val !== null && val !== undefined);
          if (values.length > 0) {
            scales.push(Math.max(...values));
          }
        }
        
        if (scales.length >= 2 && Math.max(...scales) / Math.min(...scales) > 100) {
          return "dualAxisLine";
        }
      }
    }
    
    // Multi-line chart for growth data with multiple sources
    if (isGrowthData && hasMultipleSources && isTimeSeries) {
      return "multiLine";
    }
    
    // Check for time series data
    if (isTimeSeries && numColumns === 2) return "line";
    
    // Check for 4-column growth data (SourceName + TotalGeneration + GrowthMonth + GrowthPercent)
    if (numColumns === 4 && isGrowthData) {
      const hasSource = headers.some(h => h.toLowerCase().includes('source'));
      const hasGrowthMonth = headers.some(h => h.toLowerCase().includes('growthmonth'));
      const hasGrowthPercent = headers.some(h => h.toLowerCase().includes('growthpercent'));
      
      if (hasSource && hasGrowthMonth && hasGrowthPercent) {
        return "multiLine";
      }
    }
    
    // Check for categorical data with numerical values (like SourceName + TotalGeneration)
    if (numColumns === 2 && firstColValues.length > 0) {
      const isCategorical = typeof firstColValues[0] === 'string';
      if (isCategorical) {
        // Check if second column is numerical (like TotalGeneration, SharePercentage)
        const secondColValues = rows.map(row => row[1]).filter(val => val !== null && val !== undefined);
        const isNumerical = secondColValues.length > 0 && !isNaN(parseFloat(secondColValues[0]));
        
        if (isNumerical) {
          // For data like SourceName + TotalGeneration, prefer pie chart for small datasets
          if (rows.length <= 8) return "pie";
          return "bar";
        }
        
        // Fallback for categorical data
        if (rows.length <= 7) return "pie";
        return "bar";
      }
    }
    
    // Check for 3-column data that could be grouped
    if (numColumns === 3 && firstColValues.length > 0) {
      const isCategorical = typeof firstColValues[0] === 'string';
      if (isCategorical) {
        const secondColValues = rows.map(row => row[1]).filter(val => val !== null && val !== undefined);
        if (secondColValues.length > 0 && typeof secondColValues[0] === 'string') {
          // For growth data, prefer stacked bar
          if (isGrowthData) return "stackedBar";
          return "groupedBar";
        }
      }
    }
    
    // Default fallback
    return "bar";
  }

  // Helper to guess headers from SQL if not provided
  function extractHeadersFromSQL(sql) {
    // Try to extract column names from SELECT ... AS ... or SELECT col1, col2
    const selectMatch = sql.match(/select\s+(.*?)\s+from/i);
    if (!selectMatch) return [];
    const cols = selectMatch[1]
      .split(",")
      .map((col) => {
        // Remove AS and aliases
        const asMatch = col.match(/as\s+([a-zA-Z0-9_]+)/i);
        if (asMatch) return asMatch[1].trim();
        // Remove table prefix
        return col.split(".").pop().replace(/["'`]/g, "").trim();
      })
      .filter((col) => col && col.toLowerCase() !== "distinct");
    return cols;
  }

  // Helper function to safely access plot options
  function getPlotOption(plot, option, defaultValue = '') {
    if (!plot || !plot.options) return defaultValue;
    const value = plot.options[option];
    console.log(`getPlotOption(${option}):`, value, 'default:', defaultValue);
    return value || defaultValue;
  }



  const handleCancel = () => {
    if (state.abortController) {
      state.abortController.abort();
      dispatch({ type: 'CANCEL_REQUEST' });
    }
  };

  // Health check function to verify backend availability
  const checkBackendHealth = async () => {
    try {
      const response = await fetch("http://localhost:8000/api/v1/health");
      const data = await response.json();
      return data.status === "healthy";
    } catch (error) {
      console.error("Backend health check failed:", error);
      return false;
    }
  };

  return (
    <Container maxWidth="md" style={{ marginTop: 32 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h4" gutterBottom>
          Chat with Power Data DB
        </Typography>
        
        {/* Backend Status Indicator */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography variant="caption" color="text.secondary">
            Backend:
          </Typography>
          <Chip
            label={
              state.backendStatus === 'healthy' ? '🟢 Online' :
              state.backendStatus === 'unhealthy' ? '🔴 Offline' :
              state.backendStatus === 'checking' ? '🟡 Checking...' : '⚪ Unknown'
            }
            size="small"
            color={
              state.backendStatus === 'healthy' ? 'success' :
              state.backendStatus === 'unhealthy' ? 'error' :
              'default'
            }
            variant="outlined"
          />
        </Box>
      </Box>
      
      {/* Conversation History */}
      {state.history.length > 0 && (
        <Paper style={{ padding: 16, marginBottom: 16 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">📝 Recent Questions</Typography>
            <Button 
              variant="outlined" 
              size="small" 
              onClick={() => dispatch({ type: 'CLEAR_CONVERSATION' })}
              color="warning"
            >
              Clear History
            </Button>
          </Box>
          <Stack spacing={1}>
            {state.history.slice(-3).map((item, i) => (
              <Chip
                key={i}
                label={item.question}
                variant="outlined"
                size="small"
                style={{ maxWidth: '100%', textAlign: 'left' }}
              />
            ))}
          </Stack>
        </Paper>
      )}
      <Paper style={{ padding: 16, marginBottom: 24 }}>
        <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
        <TextField
            label="Ask a question"
          fullWidth
            value={listening ? transcript : state.question}
            onChange={(e) => dispatch({ type: 'SET_FIELD', field: 'question', payload: e.target.value })}
            onKeyDown={(e) => e.key === "Enter" && handleAsk()}
            placeholder={listening ? "Listening... (speak now)" : "Type or speak your question... (will auto-submit)"}
            disabled={listening}
          />
          <Tooltip title={listening ? "Stop listening" : "Start voice input"}>
            <IconButton
              onClick={() => {
                if (listening) {
                  SpeechRecognition.stopListening();
                } else {
                  resetTranscript();
                  SpeechRecognition.startListening({ 
                    continuous: true,
                    language: 'en-US',
                    interimResults: true
                  });
                }
              }}
              color={listening ? "error" : "primary"}
              disabled={!browserSupportsSpeechRecognition}
              sx={{ 
                mt: 1,
                backgroundColor: listening ? 'error.light' : 'primary.light',
                '&:hover': {
                  backgroundColor: listening ? 'error.main' : 'primary.main',
                }
              }}
            >
              {listening ? <MicOffIcon /> : <MicIcon />}
            </IconButton>
          </Tooltip>
        </Box>
        
        {/* Voice input instructions */}
        {!listening && (
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            💡 Tip: Click the microphone and speak your question. It will automatically submit after you stop speaking.
          </Typography>
        )}
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mt: 2 }}>
          <FormControl style={{ minWidth: 120 }}>
            <InputLabel id="llm-select-label">LLM</InputLabel>
            <Select
              labelId="llm-select-label"
              value={state.llm}
              label="LLM"
              onChange={e => dispatch({ type: 'SET_FIELD', field: 'llm', payload: e.target.value })}
              style={{ minWidth: 120 }}
            >
              <MenuItem value="ollama">Ollama</MenuItem>
              <MenuItem value="openai">OpenAI</MenuItem>
              {/* Add more as needed */}
            </Select>
          </FormControl>
          
        <Button
          variant="contained"
            color="primary"
            onClick={handleAsk}
            disabled={state.loading || listening}
        >
          Ask
        </Button>
          
          {listening && (
            <Chip 
              label="🎤 Listening..." 
              color="error" 
              variant="outlined"
              icon={<MicIcon />}
              sx={{
                animation: 'pulse 1.5s ease-in-out infinite',
                '@keyframes pulse': {
                  '0%': { opacity: 1 },
                  '50%': { opacity: 0.7 },
                  '100%': { opacity: 1 },
                }
              }}
            />
          )}
          {!listening && transcript && transcript.trim().length > 0 && (
            <Chip 
              label={`⏳ Auto-submitting in 1.5s... (${transcript.trim().length} chars)`}
              color="info" 
              variant="outlined"
              sx={{
                animation: 'pulse 1s ease-in-out infinite',
                '@keyframes pulse': {
                  '0%': { opacity: 1 },
                  '50%': { opacity: 0.5 },
                  '100%': { opacity: 1 },
                }
              }}
            />
          )}
        </Box>
      </Paper>
      {state.loading && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
          <CircularProgress size={24} />
          <Typography variant="body2" color="text.secondary">
            Processing your question...
            </Typography>
          <Button 
            variant="outlined" 
            size="small" 
            onClick={handleCancel}
            color="error"
            sx={{ ml: 'auto' }}
          >
            Cancel
          </Button>
        </Box>
      )}
      {state.error && <Alert severity="error">{state.error}</Alert>}
      
      {/* Debug Info */}
      {process.env.NODE_ENV === 'development' && (
        <Paper style={{ padding: 8, marginBottom: 16, backgroundColor: '#f5f5f5' }}>
          <Typography variant="caption" color="text.secondary">
            Debug: Current Question: "{state.question}" | History Count: {state.history.length}
            </Typography>
        </Paper>
      )}
      
      {/* Clarification needed */}
      {state.clarificationNeeded && (
        <Paper style={{ padding: 16, marginBottom: 24, backgroundColor: '#fff3e0' }}>
          <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
            <Typography variant="h6" color="warning.main" sx={{ mt: 0.5 }}>
              🤔
            </Typography>
            <Box sx={{ flex: 1 }}>
              <Typography variant="h6" color="warning.main" gutterBottom>
                Clarification Needed
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                <strong>Original Question:</strong> {state.question}
              </Typography>
              <Typography variant="body1" sx={{ mb: 2 }}>
                {state.clarificationQuestion}
              </Typography>
              
              {/* Clarification Questions */}
              {state.suggestions && state.suggestions.length > 0 && (
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 'bold' }}>
                    Please answer the following questions:
                  </Typography>
                  {state.suggestions.map((question, index) => (
                    <Box key={index} sx={{ mb: 2 }}>
                      <Typography variant="body2" sx={{ mb: 1, fontWeight: 'medium' }}>
                        {index + 1}. {question}
                      </Typography>
                      <TextField
                        fullWidth
                        size="small"
                        placeholder="Type your answer here..."
                        value={state.clarificationAnswers?.[question] || ''}
                        onChange={(e) => dispatch({ 
                          type: 'SET_CLARIFICATION_ANSWER', 
                          payload: { question, answer: e.target.value }
                        })}
                        sx={{ 
                          backgroundColor: 'white',
                          '& .MuiOutlinedInput-root': {
                            '& fieldset': {
                              borderColor: 'rgba(0, 0, 0, 0.23)',
                            },
                            '&:hover fieldset': {
                              borderColor: 'rgba(0, 0, 0, 0.87)',
                            },
                          }
                        }}
                      />
                    </Box>
                  ))}
                </Box>
              )}
              
              {/* Action Buttons */}
              <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={handleAsk}
                  disabled={state.loading}
                  sx={{ minWidth: 120 }}
                >
                  {state.loading ? (
                    <>
                      <CircularProgress size={16} sx={{ mr: 1 }} />
                      Processing...
                    </>
                  ) : (
                    'Submit Answers'
                  )}
                </Button>
                <Button
                  variant="outlined"
                  color="secondary"
                  onClick={() => dispatch({ type: 'CLEAR_CLARIFICATION' })}
                  disabled={state.loading}
                >
                  Cancel
                </Button>
                <Typography variant="caption" color="text.secondary">
                  Your answers will help me provide a more accurate response.
                </Typography>
              </Box>
            </Box>
          </Box>
        </Paper>
      )}
      
      {/* Clarification Question Display */}
      {state.clarification_needed && state.clarification_question && (
        <Paper style={{ padding: 16, marginBottom: 16, backgroundColor: '#fff3cd', border: '1px solid #ffeaa7' }}>
          <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
            <Typography variant="h6" color="warning.main" sx={{ mt: 0.5 }}>
              🤔
            </Typography>
            <Box sx={{ flex: 1 }}>
              <Typography variant="h6" color="warning.main" gutterBottom>
                Clarification Needed
              </Typography>
              <Typography variant="body1" sx={{ mb: 2 }}>
                {state.clarification_question}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Please provide more specific details to help generate an accurate response.
              </Typography>
              <Button 
                variant="contained" 
                color="primary"
                onClick={() => {
                  dispatch({ type: 'SET_FIELD', field: 'clarification_needed', payload: false });
                  dispatch({ type: 'SET_FIELD', field: 'clarification_question', payload: "" });
                }}
              >
                Got it, I'll be more specific
              </Button>
            </Box>
          </Box>
        </Paper>
      )}
      
      {/* Browser compatibility warning for speech recognition */}
      {!browserSupportsSpeechRecognition && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          Speech recognition is not supported in this browser. Please use Chrome, Edge, or Safari for voice input functionality.
        </Alert>
      )}
      
      {/* Chart and Table View with Tabs */}
      {(() => {
        console.log('Chart rendering condition check:', {
          hasPlot: !!state.plot,
          hasTable: !!state.table,
          hasChartData: !!(state.table && state.table.chartData),
          chartDataLength: state.table?.chartData?.length || 0,
          plotData: state.plot,
          tableData: state.table
        });
        
        return state.plot && state.table && state.table.chartData && state.table.chartData.length > 0;
      })() && (
        <Paper style={{ padding: 16, marginBottom: 24 }}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider', marginBottom: 2 }}>
            <Tabs value={state.viewMode} onChange={(e, newValue) => dispatch({ type: 'SET_FIELD', field: 'viewMode', payload: newValue })}>
              <Tab label="Chart View" value="chart" />
              <Tab label="Table View" value="table" />
            </Tabs>
          </Box>
          
          {state.viewMode === "chart" && (
            <Box>
              <Typography variant="h6" gutterBottom>{state.plot?.options?.title || state.plot?.title || 'Data Visualization'}</Typography>
              
              {/* Chart Builder Interface */}
              {state.showChartBuilder && (
                <ChartBuilder state={state} dispatch={dispatch} />
              )}
              
              {/* Chart Type Selector and Chart Builder */}
              <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
                <Typography variant="body2" color="text.secondary">Chart Type:</Typography>
                <FormControl size="small" sx={{ minWidth: 120 }}>
                  <Select
                    value={state.selectedChartType}
                    onChange={(e) => dispatch({ type: 'SET_FIELD', field: 'selectedChartType', payload: e.target.value })}
                    displayEmpty
                  >
                    <MenuItem value="auto">Auto (AI Recommended)</MenuItem>
                    <MenuItem value="multiLine">Multi-Line Chart</MenuItem>
                    <MenuItem value="line">Line Chart</MenuItem>
                    <MenuItem value="bar">Bar Chart</MenuItem>
                    <MenuItem value="stackedBar">Stacked Bar Chart</MenuItem>
                    <MenuItem value="groupedBar">Grouped Bar Chart</MenuItem>
                    <MenuItem value="pie">Pie Chart</MenuItem>
                  </Select>
                </FormControl>
                
                {/* Chart Builder Button */}
                <Button
                  variant="outlined"
                  size="small"
                  onClick={() => dispatch({ type: 'TOGGLE_CHART_BUILDER' })}
                  startIcon={<DragIndicatorIcon />}
                >
                  🎨 Chart Builder
                </Button>
                
                {/* AI Visualization Info */}
                {state.plot?.options?.dataType && (
                  <Chip 
                    label={`AI: ${state.plot.options.dataType}`}
                    size="small" 
                    color="info" 
                    variant="outlined"
                  />
                )}
                {(() => {
                  const numColumns = state.table.headers?.length || 0;
                  const isTimeSeries = state.table.headers?.[0]?.toLowerCase().includes('date') || 
                                      state.table.headers?.[0]?.toLowerCase().includes('month') || 
                                      state.table.headers?.[0]?.toLowerCase().includes('year') ||
                                      state.table.headers?.[0]?.toLowerCase().includes('quarter');
                  
                  if (state.selectedChartType === "line" && !isTimeSeries) {
                    return (
                      <Chip 
                        label="Switched to Bar Chart (not time series data)" 
                        size="small" 
                        color="warning" 
                        variant="outlined"
                      />
                    );
                  }
                  if (state.selectedChartType === "pie" && (numColumns !== 2 || state.table.rows?.length > 10)) {
                    return (
                      <Chip 
                        label="Switched to Bar Chart (too many categories)" 
                        size="small" 
                        color="warning" 
                        variant="outlined"
                      />
                    );
                  }
                  if (state.selectedChartType === "groupedBar" && numColumns !== 3) {
                    return (
                      <Chip 
                        label="Switched to Bar Chart (need 3 columns for grouped bar)" 
                        size="small" 
                        color="warning" 
                        variant="outlined"
                      />
                    );
                  }
                  return null;
                })()}
              </Box>
              
              <ResponsiveContainer width="100%" height={400}>
                {(() => {
                  // Safety check for data
                  if (!state.table || !state.table.chartData || state.table.chartData.length === 0) {
                    return (
                      <Box sx={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'center', 
                        height: '100%',
                        color: 'text.secondary'
                      }}>
                        <Typography variant="body1">
                          No chart data available
                        </Typography>
                      </Box>
                    );
                  }
                  
                  // Determine which chart type to show
                  let chartTypeToShow = state.selectedChartType;
                  if (state.selectedChartType === "auto") {
                    chartTypeToShow = state.plot?.chartType || getChartTypeForData(state.table.headers, state.table.rows, state.plot);
                  }
                  
                  // Validate if the selected chart type is appropriate for the data
                  const numColumns = state.table.headers?.length || 0;
                  const isTimeSeries = state.table.headers?.[0]?.toLowerCase().includes('date') || 
                                      state.table.headers?.[0]?.toLowerCase().includes('month') || 
                                      state.table.headers?.[0]?.toLowerCase().includes('year') ||
                                      state.table.headers?.[0]?.toLowerCase().includes('quarter');
                  
                  // Override if selection is inappropriate
                  if (chartTypeToShow === "line" && !isTimeSeries) {
                    chartTypeToShow = "bar";
                  }
                  if (chartTypeToShow === "pie" && (numColumns !== 2 || state.table.rows?.length > 10)) {
                    chartTypeToShow = "bar";
                  }
                  if (chartTypeToShow === "groupedBar" && numColumns !== 3) {
                    chartTypeToShow = "bar";
                  }
                  
                  switch (chartTypeToShow) {
                    case 'dualAxisBarLine':
                      // Dual-axis chart with bars and lines
                      const xAxisBarLine = getPlotOption(state.plot, 'xAxis') || state.table.headers[0];
                      const chartDataBarLine = state.table.chartData;
                      
                      // Try to identify the correct columns for dual-axis
                      const headers = state.table.headers || [];
                      const energyCol = headers.find(h => h.toLowerCase().includes('energy') && h.toLowerCase().includes('total'));
                      const growthCol = headers.find(h => h.toLowerCase().includes('growth') || h.toLowerCase().includes('percent'));
                      
                      const primaryYAxis = state.plot?.options?.yAxis?.[0] || energyCol || headers[1];
                      const secondaryYAxis = state.plot?.options?.yAxisSecondary?.[0] || growthCol || headers[2];
                      
                      console.log('Dual-axis chart config:', {
                        xAxis: xAxisBarLine,
                        primaryYAxis,
                        secondaryYAxis,
                        headers,
                        plotOptions: state.plot?.options
                      });
                      
                      return (
                        <ComposedChart data={chartDataBarLine}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey={xAxisBarLine} />
                          <YAxis yAxisId="left" />
                          <YAxis yAxisId="right" orientation="right" />
                          <RechartsTooltip />
                          <Legend />
                          <Bar 
                            yAxisId="left"
                            dataKey={primaryYAxis} 
                            fill="#82ca9d"
                            name={primaryYAxis}
                          />
                          <Line 
                            yAxisId="right"
                            type="monotone" 
                            dataKey={secondaryYAxis} 
                            stroke="#ff7300" 
                            strokeWidth={2}
                            name={secondaryYAxis}
                          />
                        </ComposedChart>
                      );
                    case 'dualAxisLine':
                      // Dual-axis chart with multiple lines
                      const xAxisLine = getPlotOption(state.plot, 'xAxis') || state.table.headers[0];
                      const chartDataLine = state.table.chartData;
                      const line1 = state.plot?.options?.yAxis?.[0] || state.table.headers[1];
                      const line2 = state.plot?.options?.yAxisSecondary?.[0] || state.table.headers[2];
                      
                      return (
                        <LineChart data={chartDataLine}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey={xAxisLine} />
                          <YAxis yAxisId="left" />
                          <YAxis yAxisId="right" orientation="right" />
                          <RechartsTooltip />
                          <Legend />
                          <Line 
                            yAxisId="left"
                            type="monotone" 
                            dataKey={line1} 
                            stroke="#8884d8" 
                            strokeWidth={2}
                            name={line1}
                          />
                          <Line 
                            yAxisId="right"
                            type="monotone" 
                            dataKey={line2} 
                            stroke="#ff7300" 
                            strokeWidth={2}
                            name={line2}
                          />
                        </LineChart>
                      );
                    case 'line':
                      return (
                        <LineChart data={state.table.chartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey={getPlotOption(state.plot, 'xAxis')} />
                          <YAxis />
                          <RechartsTooltip />
                          <Legend />
                          <Line type="monotone" dataKey={getPlotOption(state.plot, 'yAxis')[0] || getPlotOption(state.plot, 'yAxis')} stroke="#8884d8" strokeWidth={2} />
                        </LineChart>
                      );
                    case 'bar':
                      return (
                        <BarChart data={state.table.chartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey={getPlotOption(state.plot, 'xAxis')} />
                          <YAxis />
                          <RechartsTooltip />
                          <Legend />
                          <Bar 
                            dataKey={getPlotOption(state.plot, 'yAxis')[0] || getPlotOption(state.plot, 'yAxis')} 
                            fill="#82ca9d"
                            onClick={(data) => {
                              const category = data[getPlotOption(state.plot, 'xAxis')];
                              const newQuestion = `Show me a detailed breakdown for ${category}`;
                              dispatch({ type: 'SET_FIELD', field: 'question', payload: newQuestion });
                            }}
                            style={{ cursor: 'pointer' }}
                          />
                        </BarChart>
                      );
                    case 'pie':
                      return (
                        <PieChart>
                          <Pie 
                            data={state.table.chartData} 
                            dataKey={getPlotOption(state.plot, 'yAxis')[0] || getPlotOption(state.plot, 'yAxis')} 
                            nameKey={getPlotOption(state.plot, 'xAxis')} 
                            cx="50%" 
                            cy="50%" 
                            outerRadius={120} 
                            fill="#8884d8" 
                            label={({name, percent}) => `${name}: ${(percent * 100).toFixed(0)}%`}
                            onClick={(data) => {
                              const category = data[getPlotOption(state.plot, 'xAxis')];
                              const newQuestion = `Show me detailed data for ${category}`;
                              dispatch({ type: 'SET_FIELD', field: 'question', payload: newQuestion });
                            }}
                            style={{ cursor: 'pointer' }}
                          />
                          <RechartsTooltip />
                        </PieChart>
                      );
                    case 'multiLine':
                      // Multi-line chart for multiple series
                      const multiLineGroupKey = getPlotOption(state.plot, 'groupBy');
                      const xAxisKey = getPlotOption(state.plot, 'xAxis');
                      const yAxisKey = getPlotOption(state.plot, 'yAxis')[0] || getPlotOption(state.plot, 'yAxis');
                      
                      if (multiLineGroupKey) {
                        const multiLineGroups = [...new Set(state.table.chartData.map(item => item[multiLineGroupKey]))];
                        const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#8dd1e1', '#d084d0', '#ff8042', '#00c49f'];
                        
                        // Transform data to group by the groupBy key and create separate data series
                        const transformedData = {};
                        state.table.chartData.forEach(item => {
                          const groupValue = item[multiLineGroupKey];
                          const xValue = item[xAxisKey];
                          const yValue = item[yAxisKey];
                          
                          if (!transformedData[xValue]) {
                            transformedData[xValue] = { [xAxisKey]: xValue };
                          }
                          transformedData[xValue][groupValue] = yValue;
                        });
                        
                        const finalChartData = Object.values(transformedData).sort((a, b) => a[xAxisKey] - b[xAxisKey]);
                        
                        return (
                          <LineChart data={finalChartData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey={xAxisKey} />
                            <YAxis />
                            <RechartsTooltip />
                            <Legend />
                            {multiLineGroups.map((group, i) => (
                              <Line 
                                key={group}
                                type="monotone"
                                dataKey={group}
                                stroke={colors[i % colors.length]}
                                strokeWidth={2}
                                name={group}
                                connectNulls
                                dot={{ r: 4 }}
                              />
                            ))}
                          </LineChart>
                        );
                      } else {
                        // Fallback to single line
                        return (
                          <LineChart data={state.table.chartData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey={xAxisKey} />
                            <YAxis />
                            <RechartsTooltip />
                            <Legend />
                            <Line 
                              type="monotone" 
                              dataKey={yAxisKey} 
                              stroke="#8884d8" 
                              strokeWidth={2} 
                            />
                          </LineChart>
                        );
                      }
                    case 'stackedBar':
                      // Stacked bar chart
                      const stackedGroupKey = getPlotOption(state.plot, 'groupBy');
                      if (stackedGroupKey) {
                        const stackedGroups = [...new Set(state.table.chartData.map(item => item[stackedGroupKey]))];
                        const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#8dd1e1', '#d084d0', '#ff8042', '#00c49f'];
                        
                        return (
                          <BarChart data={state.table.chartData} stackOffset="expand">
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey={getPlotOption(state.plot, 'xAxis')} />
                            <YAxis />
                            <RechartsTooltip />
                            <Legend />
                            {stackedGroups.map((group, i) => (
                              <Bar 
                                key={group}
                                dataKey={getPlotOption(state.plot, 'yAxis')[0] || getPlotOption(state.plot, 'yAxis')}
                                stackId="a"
                                fill={colors[i % colors.length]}
                                name={group}
                              />
                            ))}
                          </BarChart>
                        );
                      } else {
                        // Fallback to regular bar
                        return (
                          <BarChart data={state.table.chartData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey={getPlotOption(state.plot, 'xAxis')} />
                            <YAxis />
                            <RechartsTooltip />
                            <Legend />
                            <Bar 
                              dataKey={getPlotOption(state.plot, 'yAxis')[0] || getPlotOption(state.plot, 'yAxis')} 
                              fill="#82ca9d"
                            />
                          </BarChart>
                        );
                      }
                    case 'groupedBar':
                      // Transform data for grouped bar chart
                      const groupKey = getPlotOption(state.plot, 'groupBy');
                      const uniqueGroups = [...new Set(state.table.chartData.map(item => item[groupKey]))];
                      
                      // Pivot the data
                      const pivotedData = {};
                      state.table.chartData.forEach(item => {
                        const xAxisValue = item[getPlotOption(state.plot, 'xAxis')];
                        if (!pivotedData[xAxisValue]) {
                          pivotedData[xAxisValue] = { [getPlotOption(state.plot, 'xAxis')]: xAxisValue };
                        }
                        pivotedData[xAxisValue][item[groupKey]] = item[getPlotOption(state.plot, 'yAxis')[0] || getPlotOption(state.plot, 'yAxis')];
                      });
                      const finalChartData = Object.values(pivotedData);
                      
                      // Generate colors for each group
                      const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#8dd1e1', '#d084d0', '#ff8042', '#00c49f'];
                      
                      return (
                        <BarChart data={finalChartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey={getPlotOption(state.plot, 'xAxis')} />
                          <YAxis />
                          <RechartsTooltip />
                          <Legend />
                          {uniqueGroups.map((group, i) => (
                            <Bar 
                              key={group} 
                              dataKey={group} 
                              fill={colors[i % colors.length]}
                              onClick={(data) => {
                                const category = data[getPlotOption(state.plot, 'xAxis')];
                                const newQuestion = `Show me detailed data for ${category} and ${group}`;
                                dispatch({ type: 'SET_FIELD', field: 'question', payload: newQuestion });
                              }}
                              style={{ cursor: 'pointer' }}
                            />
                          ))}
                        </BarChart>
                      );
                    default:
                      return (
                        <BarChart data={state.table.chartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey={getPlotOption(state.plot, 'xAxis')} />
                          <YAxis />
                          <RechartsTooltip />
                          <Legend />
                          <Bar 
                            dataKey={getPlotOption(state.plot, 'yAxis')[0] || getPlotOption(state.plot, 'yAxis')} 
                            fill="#82ca9d"
                            onClick={(data) => {
                              const category = data[getPlotOption(state.plot, 'xAxis')];
                              const newQuestion = `Show me a detailed breakdown for ${category}`;
                              dispatch({ type: 'SET_FIELD', field: 'question', payload: newQuestion });
                            }}
                            style={{ cursor: 'pointer' }}
                          />
                        </BarChart>
                      );
                  }
                })()}
              </ResponsiveContainer>
            </Box>
          )}
          
          {state.viewMode === "table" && (
            <div style={{ height: 400, width: '100%' }}>
              {state.table && state.table.headers && state.table.rows && state.table.rows.length > 0 ? (
                <DataGrid
                  rows={state.table.rows.map((row, id) => {
                    // Handle both array and object formats
                    if (Array.isArray(row)) {
                      return {
                        id,
                        ...row.reduce((acc, cell, i) => ({ 
                          ...acc, 
                          [state.table.headers[i] || `Column${i}`]: cell 
                        }), {})
                      };
                    } else {
                      return { id, ...row };
                    }
                  })}
                  columns={state.table.headers.map(header => ({ 
                    field: header, 
                    headerName: header, 
                    flex: 1,
                    sortable: true,
                    filterable: true
                  }))}
                  pageSize={10}
                  rowsPerPageOptions={[5, 10, 25]}
                  disableSelectionOnClick
                  density="compact"
                />
              ) : (
                <Box sx={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center', 
                  height: '100%',
                  color: 'text.secondary'
                }}>
                  <Typography variant="body1">
                    No data available for table view
                  </Typography>
                </Box>
              )}
            </div>
          )}
        </Paper>
      )}
      
      {/* Fallback Table View with Manual Plot Option (when no plot is available) */}
      {state.table && state.table.headers && state.table.rows && state.table.rows.length > 0 && (!state.plot || !state.table.chartData || state.table.chartData.length === 0) && (
        <Paper style={{ padding: 16, marginBottom: 24 }}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider', marginBottom: 2 }}>
            <Tabs value={state.viewMode} onChange={(e, newValue) => dispatch({ type: 'SET_FIELD', field: 'viewMode', payload: newValue })}>
              <Tab label="Table View" value="table" />
              <Tab label="Create Plot" value="manual-chart" />
            </Tabs>
          </Box>
          
          {state.viewMode === "table" && (
            <div style={{ height: 400, width: '100%' }}>
              {state.table && state.table.headers && state.table.rows && state.table.rows.length > 0 ? (
                <DataGrid
                  rows={state.table.rows.map((row, id) => {
                    // Handle both array and object formats
                    if (Array.isArray(row)) {
                      return {
                        id,
                        ...row.reduce((acc, cell, i) => ({ 
                          ...acc, 
                          [state.table.headers[i] || `Column${i}`]: cell 
                        }), {})
                      };
                    } else {
                      return { id, ...row };
                    }
                  })}
                  columns={state.table.headers.map(header => ({ 
                    field: header, 
                    headerName: header, 
                    flex: 1,
                    sortable: true,
                    filterable: true
                  }))}
                  pageSize={10}
                  rowsPerPageOptions={[5, 10, 25]}
                  disableSelectionOnClick
                  density="compact"
                />
              ) : (
                <Box sx={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center', 
                  height: '100%',
                  color: 'text.secondary'
                }}>
                  <Typography variant="body1">
                    No data available for table view
                  </Typography>
                </Box>
              )}
            </div>
          )}
          
          {state.viewMode === "manual-chart" && (
            <Box>
              <Typography variant="h6" gutterBottom>Manual Chart Creation</Typography>
              
              {/* Chart Type Selector for Manual Plot */}
              <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
                <Typography variant="body2" color="text.secondary">Chart Type:</Typography>
                <FormControl size="small" sx={{ minWidth: 120 }}>
                  <Select
                    value={state.selectedChartType}
                    onChange={(e) => dispatch({ type: 'SET_FIELD', field: 'selectedChartType', payload: e.target.value })}
                    displayEmpty
                  >
                    <MenuItem value="auto">Auto (Recommended)</MenuItem>
                    <MenuItem value="line">Line Chart</MenuItem>
                    <MenuItem value="bar">Bar Chart</MenuItem>
                    <MenuItem value="pie">Pie Chart</MenuItem>
                    <MenuItem value="groupedBar">Grouped Bar Chart</MenuItem>
                  </Select>
                </FormControl>
                {(() => {
                  const numColumns = state.table.headers?.length || 0;
                  const isTimeSeries = state.table.headers?.[0]?.toLowerCase().includes('date') || 
                                      state.table.headers?.[0]?.toLowerCase().includes('month') || 
                                      state.table.headers?.[0]?.toLowerCase().includes('year') ||
                                      state.table.headers?.[0]?.toLowerCase().includes('quarter');
                  
                  if (state.selectedChartType === "line" && !isTimeSeries) {
                    return (
                      <Chip 
                        label="Switched to Bar Chart (not time series data)" 
                        size="small" 
                        color="warning" 
                        variant="outlined"
                      />
                    );
                  }
                  if (state.selectedChartType === "pie" && (numColumns !== 2 || state.table.rows?.length > 10)) {
                    return (
                      <Chip 
                        label="Switched to Bar Chart (too many categories)" 
                        size="small" 
                        color="warning" 
                        variant="outlined"
                      />
                    );
                  }
                  if (state.selectedChartType === "groupedBar" && numColumns !== 3) {
                    return (
                      <Chip 
                        label="Switched to Bar Chart (need 3 columns for grouped bar)" 
                        size="small" 
                        color="warning" 
                        variant="outlined"
                      />
                    );
                  }
                  return null;
                })()}
              </Box>
              
              <ResponsiveContainer width="100%" height={400}>
                {(() => {
                  // Create chart data for manual plotting
                  const chartData = state.table.rows.map(row => {
                    let obj = {};
                    state.table.headers.forEach((header, i) => {
                      obj[header] = row[i];
                    });
                    return obj;
                  });
                  
                  // Determine which chart type to show
                  let chartTypeToShow = state.selectedChartType;
                  if (state.selectedChartType === "auto") {
                    chartTypeToShow = getChartTypeForData(state.table.headers, state.table.rows);
                  }
                  
                  // Validate if the selected chart type is appropriate for the data
                  const numColumns = state.table.headers?.length || 0;
                  const isTimeSeries = state.table.headers?.[0]?.toLowerCase().includes('date') || 
                                      state.table.headers?.[0]?.toLowerCase().includes('month') || 
                                      state.table.headers?.[0]?.toLowerCase().includes('year') ||
                                      state.table.headers?.[0]?.toLowerCase().includes('quarter');
                  
                  // Override if selection is inappropriate
                  if (chartTypeToShow === "line" && !isTimeSeries) {
                    chartTypeToShow = "bar";
                  }
                  if (chartTypeToShow === "pie" && (numColumns !== 2 || state.table.rows?.length > 10)) {
                    chartTypeToShow = "bar";
                  }
                  if (chartTypeToShow === "groupedBar" && numColumns !== 3) {
                    chartTypeToShow = "bar";
                  }
                  
                  const xAxis = state.table.headers[0];
                  const yAxis = state.table.headers[1];
                  
                  switch (chartTypeToShow) {
                    case 'dualAxisBarLine':
                      // Dual-axis chart with bars and lines
                      const primaryYAxis = state.plot?.options?.yAxis?.[0] || yAxis;
                      const secondaryYAxis = state.plot?.options?.yAxisSecondary?.[0] || state.table.headers[2];
                      
                      return (
                        <ComposedChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey={xAxis} />
                          <YAxis yAxisId="left" />
                          <YAxis yAxisId="right" orientation="right" />
                          <RechartsTooltip />
                          <Legend />
                          <Bar 
                            yAxisId="left"
                            dataKey={primaryYAxis} 
                            fill="#82ca9d"
                            name={primaryYAxis}
                          />
                          <Line 
                            yAxisId="right"
                            type="monotone" 
                            dataKey={secondaryYAxis} 
                            stroke="#ff7300" 
                            strokeWidth={2}
                            name={secondaryYAxis}
                          />
                        </ComposedChart>
                      );
                    case 'dualAxisLine':
                      // Dual-axis chart with multiple lines
                      const line1 = state.plot?.options?.yAxis?.[0] || yAxis;
                      const line2 = state.plot?.options?.yAxisSecondary?.[0] || state.table.headers[2];
                      
                      return (
                        <LineChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey={xAxis} />
                          <YAxis yAxisId="left" />
                          <YAxis yAxisId="right" orientation="right" />
                          <RechartsTooltip />
                          <Legend />
                          <Line 
                            yAxisId="left"
                            type="monotone" 
                            dataKey={line1} 
                            stroke="#8884d8" 
                            strokeWidth={2}
                            name={line1}
                          />
                          <Line 
                            yAxisId="right"
                            type="monotone" 
                            dataKey={line2} 
                            stroke="#ff7300" 
                            strokeWidth={2}
                            name={line2}
                          />
                        </LineChart>
                      );
                    case 'line':
                      return (
                        <LineChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey={xAxis} />
                          <YAxis />
                          <RechartsTooltip />
                          <Legend />
                          <Line type="monotone" dataKey={yAxis} stroke="#8884d8" strokeWidth={2} />
                        </LineChart>
                      );
                    case 'bar':
                      return (
                        <BarChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey={xAxis} />
                          <YAxis />
                          <RechartsTooltip />
                          <Legend />
                          <Bar 
                            dataKey={yAxis} 
                            fill="#82ca9d"
                            onClick={(data) => {
                              const category = data[xAxis];
                              const newQuestion = `Show me a detailed breakdown for ${category}`;
                              dispatch({ type: 'SET_FIELD', field: 'question', payload: newQuestion });
                            }}
                            style={{ cursor: 'pointer' }}
                          />
                        </BarChart>
                      );
                    case 'pie':
                      return (
                        <PieChart>
                          <Pie 
                            data={chartData} 
                            dataKey={yAxis} 
                            nameKey={xAxis} 
                            cx="50%" 
                            cy="50%" 
                            outerRadius={120} 
                            fill="#8884d8" 
                            label={({name, percent}) => `${name}: ${(percent * 100).toFixed(0)}%`}
                            onClick={(data) => {
                              const category = data[xAxis];
                              const newQuestion = `Show me detailed data for ${category}`;
                              dispatch({ type: 'SET_FIELD', field: 'question', payload: newQuestion });
                            }}
                            style={{ cursor: 'pointer' }}
                          />
                          <RechartsTooltip />
                        </PieChart>
                      );
                    case 'groupedBar':
                      // Transform data for grouped bar chart
                      const groupKey = state.table.headers[1]; // Assume second column is group by
                      const uniqueGroups = [...new Set(chartData.map(item => item[groupKey]))];
                      // Pivot the data
                      const pivotedData = {};
                      chartData.forEach(item => {
                        const xAxisValue = item[xAxis];
                        if (!pivotedData[xAxisValue]) {
                          pivotedData[xAxisValue] = { [xAxis]: xAxisValue };
                        }
                        pivotedData[xAxisValue][item[groupKey]] = item[yAxis];
                      });
                      const finalChartData = Object.values(pivotedData);
                      const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#8dd1e1', '#d084d0', '#ff8042', '#00c49f'];
                      return (
                        <BarChart data={finalChartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey={xAxis} />
                          <YAxis />
                          <RechartsTooltip />
                          <Legend />
                          {uniqueGroups.map((group, i) => (
                            <Bar 
                              key={group} 
                              dataKey={group} 
                              fill={colors[i % colors.length]}
                              onClick={(data) => {
                                const category = data[xAxis];
                                const newQuestion = `Show me detailed data for ${category} and ${group}`;
                                dispatch({ type: 'SET_FIELD', field: 'question', payload: newQuestion });
                              }}
                              style={{ cursor: 'pointer' }}
                            />
                          ))}
                        </BarChart>
                      );
                    default:
                      return (
                        <BarChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey={xAxis} />
                          <YAxis />
                          <RechartsTooltip />
                          <Legend />
                          <Bar 
                            dataKey={yAxis} 
                            fill="#82ca9d"
                            onClick={(data) => {
                              const category = data[xAxis];
                              const newQuestion = `Show me a detailed breakdown for ${category}`;
                              dispatch({ type: 'SET_FIELD', field: 'question', payload: newQuestion });
                            }}
                            style={{ cursor: 'pointer' }}
                          />
                        </BarChart>
                      );
                  }
                })()}
              </ResponsiveContainer>
            </Box>
          )}
        </Paper>
      )}
      
      {/* Show SQL query even if there's an error */}
      {state.sql && (
        <Paper style={{ padding: 16, marginBottom: 16 }}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6" style={{ color: '#1976d2', fontWeight: 'bold' }}>
              🔍 SQL Query Used:
            </Typography>
            <Tooltip title={copied ? "Copied!" : "Copy SQL"}>
              <IconButton
                onClick={() => {
                  navigator.clipboard.writeText(state.sql);
                  setCopied(true);
                  setTimeout(() => setCopied(false), 2000);
                }}
                size="small"
                color={copied ? "success" : "primary"}
              >
                {copied ? <CheckCircle /> : <ContentCopy />}
              </IconButton>
            </Tooltip>
          </Box>
          <Box 
            sx={{
              backgroundColor: '#f8f9fa',
              border: '1px solid #dee2e6',
              borderRadius: '8px',
              padding: '20px',
              fontFamily: '"Fira Code", "Consolas", "Monaco", "Courier New", monospace',
              fontSize: '13px',
              lineHeight: '1.6',
              overflow: 'auto',
              maxHeight: '500px',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              boxShadow: 'inset 0 1px 3px rgba(0,0,0,0.1)'
            }}
          >
            {state.sql.split('\n').map((line, index) => {
              const trimmedLine = line.trim();
              const isKeyword = trimmedLine.toUpperCase().startsWith('SELECT') || 
                               trimmedLine.toUpperCase().startsWith('FROM') || 
                               trimmedLine.toUpperCase().startsWith('WHERE') || 
                               trimmedLine.toUpperCase().startsWith('GROUP') || 
                               trimmedLine.toUpperCase().startsWith('ORDER') || 
                               trimmedLine.toUpperCase().startsWith('JOIN') ||
                               trimmedLine.toUpperCase().startsWith('LEFT') ||
                               trimmedLine.toUpperCase().startsWith('RIGHT') ||
                               trimmedLine.toUpperCase().startsWith('INNER') ||
                               trimmedLine.toUpperCase().startsWith('OUTER') ||
                               trimmedLine.toUpperCase().startsWith('ON') ||
                               trimmedLine.toUpperCase().startsWith('AS') ||
                               trimmedLine.toUpperCase().startsWith('AND') ||
                               trimmedLine.toUpperCase().startsWith('OR');
              
              const isComment = trimmedLine.startsWith('--');
              const isEmpty = trimmedLine === '';
              
              return (
                <div key={index} style={{ 
                  padding: '1px 0',
                  color: isComment ? '#6c757d' : 
                         isEmpty ? 'transparent' : 
                         isKeyword ? '#007bff' : '#212529',
                  fontWeight: isKeyword ? 'bold' : 'normal',
                  fontStyle: isComment ? 'italic' : 'normal',
                  opacity: isEmpty ? 0.3 : 1
                }}>
                  {line}
                </div>
              );
            })}
          </Box>
          <Typography variant="caption" style={{ color: '#666', marginTop: '8px', display: 'block' }}>
            💡 This SQL query was automatically generated from your natural language question
          </Typography>
        </Paper>
      )}
      
      {/* Show "No data found" only when there's actually no data */}
      {(!state.table || !state.table.rows || state.table.rows.length === 0) && !state.loading && !state.error && (
        <div>No data found.</div>
      )}
      {state.summary && (
        <Paper style={{ padding: 16, marginBottom: 24 }}>
          <ReactMarkdown>{state.summary}</ReactMarkdown>
        </Paper>
      )}
      
      {/* Follow-up Suggestions */}
      {state.suggestions.length > 0 && (
        <Paper style={{ padding: 16, marginBottom: 24 }}>
          <Typography variant="h6" gutterBottom>💡 Suggested Follow-up Questions</Typography>
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
            {state.suggestions.map((suggestion, i) => (
              <Chip
                key={i}
                label={suggestion}
                onClick={() => {
                  dispatch({ type: 'SET_FIELD', field: 'question', payload: suggestion });
                  handleAsk();
                }}
                clickable
                color="primary"
                variant="outlined"
                style={{ marginBottom: 8 }}
              />
            ))}
          </Stack>
        </Paper>
      )}
    </Container>
  );
}

export default App;