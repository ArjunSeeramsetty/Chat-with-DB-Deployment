import React from 'react';
import {
  Paper,
  Box,
  Typography,
  TextField,
  Button,
  CircularProgress,
} from '@mui/material';
import MicIcon from '@mui/icons-material/Mic';
import MicOffIcon from '@mui/icons-material/MicOff';

const ClarificationInterface = ({ 
  clarificationQuestion, 
  clarificationAnswer, 
  loading, 
  onClarificationResponse, 
  onClarificationChange, 
  onCancel,
  // Optional voice control props (hooked from parent if provided)
  listening = false,
  onStartListening = () => {},
  onStopListening = () => {}
}) => {
  return (
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
            {clarificationQuestion}
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Please provide more specific details to help generate an accurate response.
          </Typography>
          
          {/* Interactive Clarification Response */}
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-start', mb: 2 }}>
            <TextField
              fullWidth
              multiline
              rows={2}
              placeholder="Type your clarification here..."
              value={clarificationAnswer || ""}
              onChange={(e) => onClarificationChange(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  if (!loading && (clarificationAnswer || '').trim()) {
                    onClarificationResponse();
                  }
                }
              }}
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
            {/* Voice input toggle for clarification */}
            <Button
              variant="outlined"
              color={listening ? "error" : "primary"}
              onClick={listening ? onStopListening : onStartListening}
              sx={{ minWidth: 56, height: 56 }}
            >
              {listening ? <MicOffIcon /> : <MicIcon />}
            </Button>
            <Button
              variant="contained"
              color="primary"
              onClick={onClarificationResponse}
              disabled={loading || !clarificationAnswer?.trim()}
              sx={{ minWidth: 120, height: 56 }}
            >
              {loading ? (
                <>
                  <CircularProgress size={16} sx={{ mr: 1 }} />
                  Processing...
                </>
              ) : (
                'Submit'
              )}
            </Button>
          </Box>
          
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            <Button 
              variant="outlined" 
              color="secondary"
              onClick={onCancel}
            >
              Cancel
            </Button>
            <Typography variant="caption" color="text.secondary">
              Your response will help me provide a more accurate answer.
            </Typography>
          </Box>
        </Box>
      </Box>
    </Paper>
  );
};

export default ClarificationInterface; 