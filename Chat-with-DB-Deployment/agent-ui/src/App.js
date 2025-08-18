import React, { useState, useEffect, useReducer } from 'react';
import {
  Container, Typography, Paper, Box, 
  Alert, Chip, FormControl, Select, MenuItem, Button, IconButton, Tooltip
} from '@mui/material';
import { ContentCopy, CheckCircle } from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import QueryInput from './components/QueryInput';
import ClarificationInterface from './components/ClarificationInterface';
import DataVisualization from './components/DataVisualization';
import useQueryService from './hooks/useQueryService';
import { appReducer, initialState } from './reducers/appReducer';
import useClarification from './hooks/useClarification';

function App() {
  const [state, dispatch] = useReducer(appReducer, initialState);
  const [copied, setCopied] = useState(false);
  
  // Speech recognition setup
  const {
    transcript,
    listening,
    resetTranscript,
    browserSupportsSpeechRecognition
  } = useSpeechRecognition();

  // Custom hooks
  const { handleAsk, handleCancel, checkBackendHealth } = useQueryService(dispatch, state);
  const { 
    handleClarificationResponse, 
    handleClarificationChange, 
    handleClarificationCancel,
    isClarificationNeeded
  } = useClarification(dispatch, state, handleAsk);

  // Update question when transcript changes
  useEffect(() => {
    if (transcript) {
      dispatch({ type: 'SET_FIELD', field: 'question', payload: transcript });
    }
  }, [transcript]);

  // Check backend status on mount and periodically
  useEffect(() => {
    const checkBackendStatus = async () => {
      dispatch({ type: 'SET_FIELD', field: 'backendStatus', payload: 'checking' });
      await checkBackendHealth();
    };

    checkBackendStatus();
    
    // Check every 30 seconds
    const interval = setInterval(checkBackendStatus, 30000);
    
    return () => clearInterval(interval);
  }, [checkBackendHealth]);

  // Speech recognition handlers
  const handleStartListening = () => {
    resetTranscript();
    SpeechRecognition.startListening({ continuous: true });
  };

  const handleStopListening = () => {
    SpeechRecognition.stopListening();
  };

  // Event handlers
  const handleQuestionChange = (value) => {
    dispatch({ type: 'SET_FIELD', field: 'question', payload: value });
  };

  const handleViewModeChange = (newValue) => {
    dispatch({ type: 'SET_FIELD', field: 'viewMode', payload: newValue });
  };

  const handleChartTypeChange = (value) => {
    dispatch({ type: 'SET_FIELD', field: 'selectedChartType', payload: value });
  };

  const handleToggleChartBuilder = () => {
    dispatch({ type: 'TOGGLE_CHART_BUILDER' });
  };

  const handleUpdateChartConfig = (config) => {
    dispatch({ type: 'UPDATE_CHART_CONFIG', payload: config });
  };

  const handleClearError = () => {
    dispatch({ type: 'CLEAR_ERROR' });
  };

  const handleClearConversation = () => {
    dispatch({ type: 'CLEAR_CONVERSATION' });
  };

  return (
    <Container maxWidth="md" style={{ marginTop: 32 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h4" gutterBottom>
          Chat with Power Data DB
        </Typography>
        
        {/* Backend Status Indicator */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
          <Typography variant="caption" color="text.secondary">
            Backend:
          </Typography>
          <Chip
            label={
              state.backendStatus === 'healthy' ? '🟢 Healthy' :
              state.backendStatus === 'unhealthy' ? '🔴 Unhealthy' :
              state.backendStatus === 'checking' ? '🟡 Checking...' : '⚪ Unknown'
            }
            size="small"
            color={
              state.backendStatus === 'healthy' ? 'success' :
              state.backendStatus === 'unhealthy' ? 'error' : 'default'
            }
            variant="outlined"
          />
          <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
            Endpoint:
          </Typography>
          <Chip
            label={state.selectedEndpoint}
            size="small"
            variant="outlined"
            color="info"
          />
        </Box>
      </Box>

      {/* Controls: LLM Provider + Endpoint Selection */}
      <Paper style={{ padding: 16, marginBottom: 16 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
          <Typography variant="subtitle1">LLM Provider:</Typography>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <Select
              value={state.llm}
              onChange={(e) => dispatch({ type: 'SET_FIELD', field: 'llm', payload: e.target.value })}
            >
              <MenuItem value="gemini">Gemini 2.5 Flash-Lite</MenuItem>
              <MenuItem value="openai">OpenAI GPT-4</MenuItem>
              <MenuItem value="ollama">Ollama (Local)</MenuItem>
            </Select>
          </FormControl>
          <Typography variant="subtitle1" sx={{ ml: 2 }}>API Endpoint:</Typography>
          <FormControl size="small" sx={{ minWidth: 240 }}>
            <Select
              value={state.selectedEndpoint}
              onChange={(e) => dispatch({ type: 'SET_FIELD', field: 'selectedEndpoint', payload: e.target.value })}
            >
              <MenuItem value="ask-fixed">Traditional (/api/v1/ask-fixed)</MenuItem>
              <MenuItem value="ask-enhanced">Enhanced (/api/v1/ask-enhanced)</MenuItem>
              <MenuItem value="ask">Backend Default (/api/v1/ask → enhanced)</MenuItem>
              <MenuItem value="ask-agentic">Agentic Workflow (/api/v1/ask-agentic)</MenuItem>
            </Select>
          </FormControl>
        </Box>
      </Paper>

      {/* Error Display */}
      {state.error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={handleClearError}>
          {state.error}
        </Alert>
      )}

      {/* Query Input Component */}
      <QueryInput
        question={state.question}
        onQuestionChange={handleQuestionChange}
        onAsk={handleAsk}
        onCancel={handleCancel}
        loading={state.loading}
        listening={listening}
        onStartListening={handleStartListening}
        onStopListening={handleStopListening}
        browserSupportsSpeechRecognition={browserSupportsSpeechRecognition}
        backendStatus={state.backendStatus}
      />

      {/* Clarification Interface */}
      {isClarificationNeeded && (
        <ClarificationInterface
          clarificationQuestion={state.clarificationQuestion}
          clarificationAnswer={state.clarificationAnswer}
          loading={state.loading}
          onClarificationResponse={handleClarificationResponse}
          onClarificationChange={handleClarificationChange}
          onCancel={handleClarificationCancel}
          listening={listening}
          onStartListening={handleStartListening}
          onStopListening={handleStopListening}
        />
      )}

      {/* Data Visualization Component */}
      <DataVisualization
        plot={state.plot}
        table={state.table}
        viewMode={state.viewMode}
        onViewModeChange={handleViewModeChange}
        selectedChartType={state.selectedChartType}
        onChartTypeChange={handleChartTypeChange}
        showChartBuilder={state.showChartBuilder}
        onToggleChartBuilder={handleToggleChartBuilder}
        onUpdateChartConfig={handleUpdateChartConfig}
        chartConfig={state.chartConfig}
        availableColumns={state.availableColumns}
      />

      {/* SQL Query Display */}
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

      {/* Summary Display */}
      {state.summary && (
        <Paper style={{ padding: 16, marginBottom: 16 }}>
          <Typography variant="h6" gutterBottom>Summary</Typography>
          <Box>
            <ReactMarkdown>{state.summary}</ReactMarkdown>
          </Box>
        </Paper>
      )}

      {/* Suggestions */}
      {state.suggestions && state.suggestions.length > 0 && (
        <Paper style={{ padding: 16, marginBottom: 16 }}>
          <Typography variant="h6" gutterBottom>Follow-up Questions</Typography>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            {state.suggestions.map((suggestion, index) => (
              <Chip
                key={index}
                label={suggestion}
                variant="outlined"
                onClick={() => handleQuestionChange(suggestion)}
                sx={{ alignSelf: 'flex-start', cursor: 'pointer' }}
              />
            ))}
          </Box>
        </Paper>
      )}

      {/* Conversation History */}
      {state.history && state.history.length > 0 && (
        <Paper style={{ padding: 16, marginBottom: 16 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">Conversation History</Typography>
            <Button variant="outlined" size="small" onClick={handleClearConversation}>
              Clear History
            </Button>
          </Box>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {state.history.map((item, index) => (
              <Box key={index} sx={{ border: '1px solid #e0e0e0', borderRadius: 1, p: 2 }}>
                <Typography variant="subtitle2" color="primary" gutterBottom>
                  {item.question}
                </Typography>
                {item.summary && (
                  <Box sx={{ color: 'text.secondary', fontSize: '0.875rem' }}>
                    <ReactMarkdown>{item.summary}</ReactMarkdown>
                  </Box>
                )}
              </Box>
            ))}
          </Box>
        </Paper>
      )}
    </Container>
  );
}

export default App; 