import { useCallback, useMemo } from 'react';

const useClarification = (dispatch, state, handleAsk) => {
  const handleClarificationResponse = useCallback(async () => {
    const clarificationAnswer = state.clarificationAnswer || "";
    
    console.log("=== CLARIFICATION RESPONSE DEBUG ===");
    console.log("clarificationAnswer:", clarificationAnswer);
    console.log("state.clarificationQuestion:", state.clarificationQuestion);
    console.log("state.clarificationAttemptCount:", state.clarificationAttemptCount);
    
    if (!clarificationAnswer.trim()) {
      dispatch({ type: 'QUERY_ERROR', payload: "Please provide a response to the clarification question." });
      return;
    }
    
    // Store the clarification answer
    dispatch({ 
      type: 'SET_CLARIFICATION_ANSWER', 
      payload: { 
        question: state.clarificationQuestion, 
        answer: clarificationAnswer 
      } 
    });
    
    // Clear the clarification input
    dispatch({ type: 'SET_FIELD', field: 'clarificationAnswer', payload: "" });
    
    console.log("Calling handleAsk with clarification context...");
    // Continue with the original query using the clarification context
    await handleAsk();
  }, [state.clarificationAnswer, state.clarificationQuestion, state.clarificationAttemptCount, dispatch, handleAsk]);

  const handleClarificationChange = useCallback((value) => {
    dispatch({ type: 'SET_FIELD', field: 'clarificationAnswer', payload: value });
  }, [dispatch]);

  const handleClarificationCancel = useCallback(() => {
    dispatch({ type: 'SET_FIELD', field: 'clarificationNeeded', payload: false });
    dispatch({ type: 'SET_FIELD', field: 'clarificationQuestion', payload: "" });
    dispatch({ type: 'SET_FIELD', field: 'clarificationAnswer', payload: "" });
  }, [dispatch]);

  const isClarificationNeeded = useMemo(() => {
    return state.clarificationNeeded && state.clarificationQuestion;
  }, [state.clarificationNeeded, state.clarificationQuestion]);

  return {
    handleClarificationResponse,
    handleClarificationChange,
    handleClarificationCancel,
    isClarificationNeeded,
  };
};

export default useClarification; 