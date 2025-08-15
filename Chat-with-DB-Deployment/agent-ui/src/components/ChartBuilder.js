import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Box,
  Typography,
  Chip,
  TextField,
  Checkbox,
  FormControlLabel,
  Divider,
  Alert,
} from '@mui/material';

const ChartBuilder = ({
  open,
  onClose,
  onApply,
  availableColumns = [],
  currentConfig = {},
}) => {
  const [config, setConfig] = useState({
    xAxis: currentConfig.xAxis || '',
    yAxis: currentConfig.yAxis || [],
    yAxisSecondary: currentConfig.yAxisSecondary || [],
    secondaryAxis: currentConfig.secondaryAxis || '', // Single field for secondary axis
    groupBy: currentConfig.groupBy || '',
    chartType: currentConfig.chartType || 'auto',
    title: currentConfig.title || '',
    showLegend: currentConfig.showLegend !== false,
    showGrid: currentConfig.showGrid !== false,
  });

  const handleConfigChange = (field, value) => {
    setConfig(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleYAxisChange = (value) => {
    setConfig(prev => ({
      ...prev,
      yAxis: Array.isArray(value) ? value : [value]
    }));
  };

  const handleYAxisSecondaryChange = (value) => {
    setConfig(prev => ({
      ...prev,
      yAxisSecondary: Array.isArray(value) ? value : [value]
    }));
  };

  const handleSecondaryAxisChange = (value) => {
    setConfig(prev => ({
      ...prev,
      secondaryAxis: value,
      // If a single secondary axis is selected, update yAxisSecondary
      yAxisSecondary: value ? [value] : []
    }));
  };

  const handleApply = () => {
    onApply(config);
    onClose();
  };

  const chartTypes = [
    { value: 'auto', label: 'Auto (AI Recommended)' },
    { value: 'line', label: 'Line Chart' },
    { value: 'bar', label: 'Bar Chart' },
    { value: 'multiLine', label: 'Multi-Line Chart' },
    { value: 'stackedBar', label: 'Stacked Bar Chart' },
    { value: 'groupedBar', label: 'Grouped Bar Chart' },
    { value: 'pie', label: 'Pie Chart' },
  ];

  // Filter out non-numeric columns for Y-axis selection
  const numericColumns = availableColumns.filter(column => {
    // Define keywords that indicate numeric columns
    const numericKeywords = [
      'value', 'amount', 'total', 'sum', 'count', 'percentage', 'growth', 
      'current', 'previous', 'generation', 'consumption', 'shortage', 'met',
      'maximum', 'minimum', 'average', 'avg', 'max', 'min'
    ];
    
    // Define keywords that indicate non-numeric columns
    const nonNumericKeywords = [
      'name', 'region', 'state', 'source', 'trend', 'month', 'year', 'quarter', 
      'week', 'day', 'id', 'type', 'category', 'description', 'status'
    ];
    
    const columnLower = column.toLowerCase();
    
    // Check if column contains numeric keywords
    const hasNumericKeyword = numericKeywords.some(keyword => 
      columnLower.includes(keyword)
    );
    
    // Check if column contains non-numeric keywords
    const hasNonNumericKeyword = nonNumericKeywords.some(keyword => 
      columnLower.includes(keyword)
    );
    
    // Include if it has numeric keywords OR doesn't have non-numeric keywords
    return hasNumericKeyword || !hasNonNumericKeyword;
  });

  // Get all columns for X-axis and Group By
  const allColumns = availableColumns;

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        🎨 Chart Builder
        <Typography variant="body2" color="text.secondary">
          Customize your chart visualization
        </Typography>
      </DialogTitle>
      
      <DialogContent>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          
          {/* Chart Type Selection */}
          <FormControl fullWidth>
            <InputLabel>Chart Type</InputLabel>
            <Select
              value={config.chartType}
              onChange={(e) => handleConfigChange('chartType', e.target.value)}
              label="Chart Type"
            >
              {chartTypes.map(type => (
                <MenuItem key={type.value} value={type.value}>
                  {type.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* X-Axis Selection */}
          <FormControl fullWidth>
            <InputLabel>X-Axis</InputLabel>
            <Select
              value={config.xAxis}
              onChange={(e) => handleConfigChange('xAxis', e.target.value)}
              label="X-Axis"
            >
              <MenuItem value="">
                <em>Select X-Axis</em>
              </MenuItem>
              {allColumns.map(column => (
                <MenuItem key={column} value={column}>
                  {column}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <Divider />

          {/* Y-Axis Configuration */}
          <Box>
            <Typography variant="h6" gutterBottom>
              Y-Axis Configuration
            </Typography>
            
            {/* Primary Y-Axis Selection */}
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Y-Axis (Primary)</InputLabel>
              <Select
                multiple
                value={config.yAxis}
                onChange={(e) => handleYAxisChange(e.target.value)}
                label="Y-Axis (Primary)"
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => (
                      <Chip key={value} label={value} size="small" />
                    ))}
                  </Box>
                )}
              >
                {numericColumns.map(column => (
                  <MenuItem key={column} value={column}>
                    {column}
                  </MenuItem>
                ))}
              </Select>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                Select one or more columns for the primary Y-axis. These will be displayed on the left side.
              </Typography>
            </FormControl>

            {/* Secondary Y-Axis Selection */}
            <FormControl fullWidth>
              <InputLabel>Y-Axis (Secondary)</InputLabel>
              <Select
                multiple
                value={config.yAxisSecondary}
                onChange={(e) => handleYAxisSecondaryChange(e.target.value)}
                label="Y-Axis (Secondary)"
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => (
                      <Chip key={value} label={value} size="small" color="secondary" />
                    ))}
                  </Box>
                )}
              >
                <MenuItem value="">
                  <em>None</em>
                </MenuItem>
                {numericColumns.map(column => (
                  <MenuItem key={column} value={column}>
                    {column}
                  </MenuItem>
                ))}
              </Select>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                Select one or more columns for the secondary Y-axis. These will be displayed on the right side.
              </Typography>
            </FormControl>

            {/* Single Secondary Axis Field Selector */}
            <FormControl fullWidth sx={{ mt: 2 }}>
              <InputLabel>Secondary Y-Axis (Single Field)</InputLabel>
              <Select
                value={config.secondaryAxis}
                onChange={(e) => handleSecondaryAxisChange(e.target.value)}
                label="Secondary Y-Axis (Single Field)"
              >
                <MenuItem value="">
                  <em>None</em>
                </MenuItem>
                {numericColumns.map(column => (
                  <MenuItem key={column} value={column}>
                    {column}
                  </MenuItem>
                ))}
              </Select>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                Quick selection for a single field to display on the secondary axis. Useful for growth/percentage fields.
              </Typography>
            </FormControl>

            {/* Warning about overlapping columns */}
            {config.yAxis.length > 0 && config.yAxisSecondary.length > 0 && 
             config.yAxis.some(col => config.yAxisSecondary.includes(col)) && (
              <Alert severity="warning" sx={{ mt: 2 }}>
                Warning: Some columns are selected for both primary and secondary axes. 
                They will only appear on the primary axis.
              </Alert>
            )}
          </Box>

          <Divider />

          {/* Group By */}
          <FormControl fullWidth>
            <InputLabel>Group By</InputLabel>
            <Select
              value={config.groupBy}
              onChange={(e) => handleConfigChange('groupBy', e.target.value)}
              label="Group By"
            >
              <MenuItem value="">
                <em>None (Show all data points)</em>
              </MenuItem>
              {allColumns.map(column => (
                <MenuItem key={column} value={column}>
                  {column} (Aggregate by {column})
                </MenuItem>
              ))}
            </Select>
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
              Group By will aggregate your data by the selected column. For example, selecting "Month" will sum all values for each month.
            </Typography>
          </FormControl>

          {/* Chart Title */}
          <TextField
            fullWidth
            label="Chart Title"
            value={config.title}
            onChange={(e) => handleConfigChange('title', e.target.value)}
            placeholder="Enter chart title..."
          />

          {/* Chart Options */}
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Chart Options
            </Typography>
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={config.showLegend}
                    onChange={(e) => handleConfigChange('showLegend', e.target.checked)}
                  />
                }
                label="Show Legend"
              />
              <FormControlLabel
                control={
                  <Checkbox
                    checked={config.showGrid}
                    onChange={(e) => handleConfigChange('showGrid', e.target.checked)}
                  />
                }
                label="Show Grid"
              />
            </Box>
          </Box>

          {/* Available Columns Info */}
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Available Columns
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {allColumns.map(column => (
                <Chip 
                  key={column} 
                  label={column} 
                  size="small" 
                  variant="outlined"
                  color={numericColumns.includes(column) ? "primary" : "default"}
                />
              ))}
            </Box>
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
              Blue chips indicate numeric columns suitable for Y-axis
            </Typography>
          </Box>

        </Box>
      </DialogContent>
      
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={handleApply} variant="contained" color="primary">
          Apply Configuration
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ChartBuilder; 