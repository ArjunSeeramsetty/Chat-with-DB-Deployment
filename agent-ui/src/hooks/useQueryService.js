import { useCallback } from 'react';

const useQueryService = (dispatch, state) => {
  const handleAsk = useCallback(async () => {
    const currentQuestion = state.question;
    console.log("Sending question:", currentQuestion);
    
    // === SPRINT 2: CLARIFICATION LIMITS ===
    // Check if we've exceeded the maximum clarification attempts
    const maxClarificationAttempts = 3;
    if (state.clarificationAttemptCount >= maxClarificationAttempts) {
      dispatch({ 
        type: 'QUERY_ERROR', 
        payload: `Maximum clarification attempts (${maxClarificationAttempts}) reached. Please rephrase your question with more specific details.` 
      });
      return;
    }
    
    // Check if backend is healthy using cached status
    if (state.backendStatus !== 'healthy') {
      dispatch({ type: 'QUERY_ERROR', payload: "Backend is not available. Please check if the server is running." });
      return;
    }
    
    // Create abort controller for this request
    const abortController = new AbortController();
    
    dispatch({ type: 'START_QUERY', payload: { abortController } });
    
    try {
      // Prepare request payload
      const requestPayload = { 
        question: currentQuestion,
        user_id: "default_user",
        processing_mode: "balanced"
      };
      
      // If we have clarification answers, include them
      if (Object.keys(state.clarificationAnswers).length > 0) {
        console.log("=== SENDING CLARIFICATION ANSWERS ===");
        console.log("state.clarificationAnswers:", state.clarificationAnswers);
        console.log("state.clarificationAttemptCount:", state.clarificationAttemptCount);
        requestPayload.clarification_answers = state.clarificationAnswers;
        requestPayload.clarification_attempt_count = state.clarificationAttemptCount;
        console.log("Full request payload:", requestPayload);
        console.log("JSON.stringify(requestPayload):", JSON.stringify(requestPayload));
      } else {
        console.log("=== NO CLARIFICATION ANSWERS ===");
        console.log("state.clarificationAnswers:", state.clarificationAnswers);
        console.log("state.clarificationAttemptCount:", state.clarificationAttemptCount);
        console.log("state.clarificationQuestion:", state.clarificationQuestion);
        console.log("state.clarificationAnswer:", state.clarificationAnswer);
      }
      
      // Get API base URL from environment or use default
      const apiBaseUrl = process.env.REACT_APP_API_BASE_URL || "http://localhost:8000";
      
      // Send request to the API endpoint
      const response = await fetch(`${apiBaseUrl}/api/v1/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestPayload),
        signal: abortController.signal,
      });
      
      const data = await response.json();
      console.log("Received response for question:", currentQuestion);
      console.log("Response data:", data);
      
      if (!data.success) {
        // Check if this is a clarification request
        console.log("=== CLARIFICATION DEBUG ===");
        console.log("Response success:", data.success);
        console.log("Response data:", data);
        console.log("clarification_question:", data.clarification_question);
        console.log("clarification_needed:", data.clarification_needed);
        console.log("clarification_attempt_count:", data.clarification_attempt_count);
        
        if (data.clarification_question) {
          console.log("✅ Found clarification question, dispatching QUERY_SUCCESS");
          
          // === SPRINT 2: CLARIFICATION LIMIT CHECK ===
          const currentAttemptCount = data.clarification_attempt_count || 0;
          const maxClarificationAttempts = 3;
          
          if (currentAttemptCount >= maxClarificationAttempts) {
            console.log("❌ Maximum clarification attempts reached");
            dispatch({ 
              type: 'QUERY_ERROR', 
              payload: `Maximum clarification attempts (${maxClarificationAttempts}) reached. Please rephrase your question with more specific details.` 
            });
            return;
          }
          
          const payload = {
            clarification_needed: data.clarification_needed || true,
            clarification_question: data.clarification_question,
            clarification_attempt_count: currentAttemptCount,
            currentQuestion: currentQuestion,
            error: data.error || "Query needs clarification"
          };
          console.log("Dispatching payload:", payload);
          dispatch({ 
            type: 'QUERY_SUCCESS', 
            payload: payload
          });
        } else {
          console.log("❌ No clarification question found, dispatching QUERY_ERROR");
          dispatch({ type: 'QUERY_ERROR', payload: data.error || 'Unknown error occurred' });
        }
      } else {
        // Transform data for Recharts if we have table data
        let transformedTable = data.table || { headers: [], rows: [], chartData: [] };
        if (data.data && Array.isArray(data.data) && data.data.length > 0) {
          // Convert the data array to table format
          const headers = Object.keys(data.data[0]);
          const rows = data.data;
          const chartData = data.data;
          
          transformedTable = {
            headers: headers,
            rows: rows,
            chartData: chartData
          };
        }
        
        console.log('Frontend data transformation:', {
          originalData: data.data,
          originalTable: data.table,
          transformedTable: transformedTable,
          plot: data.plot,
          visualization: data.visualization,
          sql_query: data.sql_query
        });
        
        dispatch({ 
          type: 'QUERY_SUCCESS', 
          payload: {
            sql: data.sql_query || "",
            summary: data.intent_analysis ? `Query Type: ${data.query_type}, Intent: ${data.intent_analysis.intent}` : "",
            plot: data.plot || data.visualization || null,
            suggestions: data.follow_up_suggestions || [],
            table: transformedTable,
            clarification_needed: false, // Clear clarification state on success
            clarification_question: "",
            currentQuestion: currentQuestion,
            processing_mode: data.processing_mode || "balanced",
            api_calls: data.api_calls || 0,
            processing_time: data.processing_time || 0
          }
        });
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        console.log('Request was cancelled');
        // Don't dispatch error for cancelled requests
      } else {
        console.error('API Error:', err);
        dispatch({ type: 'QUERY_ERROR', payload: "Failed to connect to backend." });
      }
    }
  }, [state.question, state.backendStatus, state.clarificationAnswers, state.clarificationAnswer, state.clarificationAttemptCount, state.clarificationQuestion, dispatch]);

  const handleCancel = useCallback(() => {
    if (state.abortController) {
      state.abortController.abort();
    }
    dispatch({ type: 'CANCEL_REQUEST' });
  }, [state.abortController, dispatch]);

  const checkBackendHealth = useCallback(async () => {
    try {
      // Get API base URL from environment or use default
      const apiBaseUrl = process.env.REACT_APP_API_BASE_URL || "http://localhost:8000";
      
      const response = await fetch(`${apiBaseUrl}/api/v1/health`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      });
      
      if (response.ok) {
        dispatch({ type: 'SET_FIELD', field: 'backendStatus', payload: 'healthy' });
      } else {
        dispatch({ type: 'SET_FIELD', field: 'backendStatus', payload: 'unhealthy' });
      }
    } catch (error) {
      console.error("Backend health check failed:", error);
      dispatch({ type: 'SET_FIELD', field: 'backendStatus', payload: 'unhealthy' });
    }
  }, [dispatch]);

  return {
    handleAsk,
    handleCancel,
    checkBackendHealth,
  };
};

export default useQueryService; 