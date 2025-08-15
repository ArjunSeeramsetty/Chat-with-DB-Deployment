import React from 'react';
import {
  TextField,
  Button,
  Box,
  IconButton,
  Tooltip,
  Alert,
} from '@mui/material';
import MicIcon from '@mui/icons-material/Mic';
import MicOffIcon from '@mui/icons-material/MicOff';

const QueryInput = ({
  question,
  onQuestionChange,
  onAsk,
  onCancel,
  loading,
  listening,
  onStartListening,
  onStopListening,
  browserSupportsSpeechRecognition,
  backendStatus,
}) => {
  return (
    <Box>
      {/* Browser compatibility warning for speech recognition */}
      {!browserSupportsSpeechRecognition && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          Speech recognition is not supported in this browser. Please use Chrome, Edge, or Safari for voice input functionality.
        </Alert>
      )}

      {/* Query Input Section */}
      <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-start', mb: 2 }}>
        <TextField
          fullWidth
          multiline
          rows={3}
          placeholder="Ask a question about power data... (e.g., 'What is the energy shortage of Maharashtra in July 2025?')"
          value={question}
          onChange={(e) => onQuestionChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              if (!loading && question.trim() && backendStatus === 'healthy') {
                onAsk();
              }
            }
          }}
          disabled={loading}
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
        
        {/* Voice Input Button */}
        {browserSupportsSpeechRecognition && (
          <Tooltip title={listening ? "Stop listening" : "Start voice input"}>
            <IconButton
              onClick={listening ? onStopListening : onStartListening}
              disabled={loading}
              color={listening ? "error" : "primary"}
              sx={{ 
                backgroundColor: listening ? 'error.light' : 'primary.light',
                color: 'white',
                '&:hover': {
                  backgroundColor: listening ? 'error.main' : 'primary.main',
                }
              }}
            >
              {listening ? <MicOffIcon /> : <MicIcon />}
            </IconButton>
          </Tooltip>
        )}
      </Box>

      {/* Action Buttons */}
      <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mb: 2 }}>
        <Button
          variant="contained"
          color="primary"
          onClick={onAsk}
          disabled={loading || !question.trim() || backendStatus !== 'healthy'}
          sx={{ minWidth: 120 }}
        >
          {loading ? 'Processing...' : 'Ask Question'}
        </Button>
        
        {loading && (
          <Button
            variant="outlined"
            color="secondary"
            onClick={onCancel}
          >
            Cancel
          </Button>
        )}
        
        <Box sx={{ flex: 1 }}>
          {listening && (
            <Alert severity="info" sx={{ py: 0 }}>
              Listening... Speak now
            </Alert>
          )}
        </Box>
      </Box>
    </Box>
  );
};

export default QueryInput; 