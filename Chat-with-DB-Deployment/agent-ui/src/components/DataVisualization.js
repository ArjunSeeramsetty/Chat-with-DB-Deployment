import React from 'react';
import {
  Paper, Box, Typography, Tabs, Tab, FormControl, Select, MenuItem, 
  Button, Chip
} from '@mui/material';
import { DragIndicator as DragIndicatorIcon } from '@mui/icons-material';
import { DataGrid } from '@mui/x-data-grid';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, 
  Legend, ResponsiveContainer, LineChart, Line, PieChart, Pie, ComposedChart
} from 'recharts';
import ChartBuilder from './ChartBuilder';

const DataVisualization = ({
  plot,
  table,
  viewMode,
  onViewModeChange,
  selectedChartType,
  onChartTypeChange,
  showChartBuilder,
  onToggleChartBuilder,
  onUpdateChartConfig,
  chartConfig,
  availableColumns,
}) => {
  // State to store axis information for hover display
  const [axisInfo, setAxisInfo] = React.useState(null);

  // Debug logging for chart configuration
  React.useEffect(() => {
    if (plot) {
      console.log('Plot Configuration:', plot);
    }
    if (chartConfig) {
      console.log('Chart Config:', chartConfig);
    }
    if (table && table.chartData) {
      console.log('Table Data Sample:', table.chartData.slice(0, 2));
      console.log('Table Headers:', table.headers);
    }
  }, [plot, chartConfig, table]);

  // Helper function to calculate axis info
  const calculateAxisInfo = React.useCallback(() => {
    if (!table || !table.chartData || table.chartData.length === 0) {
      return null;
    }

    let yAxisKeys = chartConfig?.yAxis?.length > 0 
      ? chartConfig.yAxis 
      : table.headers.slice(1);
    
    // Validate that yAxisKeys actually exist in the data
    if (table && table.chartData && table.chartData.length > 0) {
      const availableFields = Object.keys(table.chartData[0]);
      yAxisKeys = yAxisKeys.filter(key => availableFields.includes(key));
      
      // If no valid yAxisKeys found, use the first numeric field from data
      if (yAxisKeys.length === 0) {
        const numericFields = availableFields.filter(field => 
          typeof table.chartData[0][field] === 'number'
        );
        yAxisKeys = numericFields.length > 0 ? numericFields : ['Value'];
      }
    }
    
    const groupBy = chartConfig?.groupBy;

    // Enhanced secondary axis detection
    const secondaryAxisKeys = chartConfig?.yAxisSecondary || [];
    const singleSecondaryAxis = chartConfig?.secondaryAxis || '';
    
    if (singleSecondaryAxis && !secondaryAxisKeys.includes(singleSecondaryAxis)) {
      secondaryAxisKeys.push(singleSecondaryAxis);
    }
    
    // Robustly determine primary and secondary Y-axis columns
    let primaryAxisKeys = [];
    let finalSecondaryAxisKeys = [];
    
    if (groupBy && yAxisKeys.length > 0) {
      // For grouped data, distribute series between primary and secondary
      const allSeries = yAxisKeys;
      
      if (secondaryAxisKeys.length > 0) {
        primaryAxisKeys = allSeries.filter(series => !secondaryAxisKeys.includes(series));
        finalSecondaryAxisKeys = allSeries.filter(series => secondaryAxisKeys.includes(series));
      } else {
        const secondaryKeywords = [
          'growth', 'percentage', 'ratio', 'rate', 'change', 'increase', 'decrease'
        ];
        
        primaryAxisKeys = allSeries.filter(series => {
          if (!series || typeof series !== 'string') return false;
          const seriesLower = series.toLowerCase();
          return !secondaryKeywords.some(keyword => seriesLower.includes(keyword));
        });
        
        finalSecondaryAxisKeys = allSeries.filter(series => {
          if (!series || typeof series !== 'string') return false;
          const seriesLower = series.toLowerCase();
          return secondaryKeywords.some(keyword => seriesLower.includes(keyword));
        });
        
        // Additional check for specific column names
        const specificSecondaryColumns = ['GrowthPercentage'];
        
        const specificSecondary = allSeries.filter(series => 
          series && typeof series === 'string' && specificSecondaryColumns.some(col => series.includes(col))
        );
        
        if (specificSecondary.length > 0) {
          finalSecondaryAxisKeys = specificSecondary;
          primaryAxisKeys = allSeries.filter(series => 
            series && typeof series === 'string' && !specificSecondaryColumns.some(col => series.includes(col))
          );
        }
      }
    } else {
      // For non-grouped data
      primaryAxisKeys = yAxisKeys.filter(key => !secondaryAxisKeys.includes(key));
      finalSecondaryAxisKeys = secondaryAxisKeys;
      
      if (finalSecondaryAxisKeys.length === 0 && yAxisKeys.length > 1) {
        const secondaryKeywords = [
          'growth', 'percentage', 'ratio', 'rate', 'change', 'increase', 'decrease'
        ];
        
        const potentialSecondary = yAxisKeys.filter(key => {
          if (!key || typeof key !== 'string') return false;
          const keyLower = key.toLowerCase();
          return secondaryKeywords.some(keyword => keyLower.includes(keyword));
        });
        
        if (potentialSecondary.length > 0) {
          finalSecondaryAxisKeys = potentialSecondary;
          primaryAxisKeys = yAxisKeys.filter(key => !potentialSecondary.includes(key));
        }
      }
    }

    // Ensure we have at least some primary axis keys
    if (primaryAxisKeys.length === 0 && yAxisKeys.length > 0) {
      primaryAxisKeys = yAxisKeys.slice(0, Math.ceil(yAxisKeys.length / 2));
      finalSecondaryAxisKeys = yAxisKeys.slice(Math.ceil(yAxisKeys.length / 2));
    }

    if (finalSecondaryAxisKeys.length > 0) {
      return { primaryAxisKeys, finalSecondaryAxisKeys };
    }
    return null;
  }, [table, chartConfig]);

  // Update axis info when dependencies change
  React.useEffect(() => {
    const newAxisInfo = calculateAxisInfo();
    setAxisInfo(newAxisInfo);
  }, [calculateAxisInfo]);

  // Helper function to get chart type for data
  const getChartTypeForData = (headers, rows, plotRecommendation) => {
    if (!headers || !rows || rows.length === 0) return 'bar';
    
    const numColumns = headers.length;
    const numRows = rows.length;
    
    // Check if it's time series data
    const isTimeSeries = headers.some(header => 
      header && typeof header === 'string' && (
        header.toLowerCase().includes('date') || 
        header.toLowerCase().includes('month') || 
        header.toLowerCase().includes('year') ||
        header.toLowerCase().includes('quarter')
      )
    );
    
    // Check if it's growth data
    const isGrowthData = plotRecommendation?.options?.dataType === "growth_time_series";
    
    // Check for dual-axis potential
    const hasGrowthColumns = headers.some(header => 
      header && typeof header === 'string' && (
        header.toLowerCase().includes('growth') || 
        header.toLowerCase().includes('percentage')
      )
    );
    const hasTotalColumns = headers.some(header => 
      header && typeof header === 'string' && (
        !header.toLowerCase().includes('growth') && 
        !header.toLowerCase().includes('percentage') &&
        (header.toLowerCase().includes('total') || 
         header.toLowerCase().includes('value') ||
         header.toLowerCase().includes('amount'))
      )
    );
    const hasDualAxisPotential = hasGrowthColumns && hasTotalColumns;
    
    if (hasDualAxisPotential && isTimeSeries) {
      return 'dualAxisLine';
    } else if (hasDualAxisPotential) {
      return 'dualAxisBarLine';
    } else if (isGrowthData) {
      return 'multiLine';
    } else if (isTimeSeries && numRows > 1) {
      return 'line';
    } else if (numColumns === 2 && numRows <= 10) {
      return 'pie';
    }
    return 'bar';
  };

  // Helper function to get chart title
  const getChartTitle = () => {
    if (!table || !table.headers || table.headers.length === 0) {
      return 'Chart';
    }
    
    const xAxisKey = chartConfig?.xAxis || table.headers[0];
    const baseTitle = chartConfig?.title || `Chart: ${xAxisKey}`;
    
    const secondaryAxisKeys = chartConfig?.yAxisSecondary || [];
    const singleSecondaryAxis = chartConfig?.secondaryAxis || '';
    const hasSecondaryAxis = secondaryAxisKeys.length > 0 || singleSecondaryAxis;
    
    if (hasSecondaryAxis) {
      return `${baseTitle} (Dual-Axis)`;
    }
    return baseTitle;
  };

  // Custom component for hoverable dual-axis info
  const HoverableDualAxisInfo = ({ primaryAxisKeys, finalSecondaryAxisKeys }) => {
    const [showInfo, setShowInfo] = React.useState(false);
    
    return (
      <Box sx={{ position: 'relative', display: 'inline-block' }}>
        <Typography
          variant="caption"
          color="primary"
          sx={{
            cursor: 'pointer',
            textDecoration: 'underline',
            '&:hover': {
              color: 'primary.dark'
            }
          }}
          onMouseEnter={() => setShowInfo(true)}
          onMouseLeave={() => setShowInfo(false)}
        >
          📊 Dual-Axis Chart
        </Typography>
        
        {showInfo && (
          <Box sx={{ 
            position: 'absolute', 
            bottom: '100%', 
            left: 0, 
            backgroundColor: 'rgba(255, 255, 255, 0.98)', 
            padding: 1.5, 
            borderRadius: 2,
            border: '1px solid #ddd',
            zIndex: 10,
            maxWidth: '350px',
            fontSize: '0.75rem',
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            mb: 1
          }}>
            <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 'bold', display: 'block', mb: 1 }}>
              📈 Chart Configuration
            </Typography>
            <Box sx={{ mb: 1 }}>
              <Typography variant="caption" color="primary" sx={{ display: 'block', mb: 0.5 }}>
                <strong>Left Axis:</strong>
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', ml: 1 }}>
                {primaryAxisKeys.slice(0, 3).join(', ')}
                {primaryAxisKeys.length > 3 && ` +${primaryAxisKeys.length - 3} more`}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="secondary" sx={{ display: 'block', mb: 0.5 }}>
                <strong>Right Axis:</strong>
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', ml: 1 }}>
                {finalSecondaryAxisKeys.slice(0, 3).join(', ')}
                {finalSecondaryAxisKeys.length > 3 && ` +${finalSecondaryAxisKeys.length - 3} more`}
              </Typography>
            </Box>
          </Box>
        )}
      </Box>
    );
  };

  // Helper function to render hoverable dual-axis info
  const renderHoverableDualAxisInfo = (primaryAxisKeys, finalSecondaryAxisKeys) => {
    if (finalSecondaryAxisKeys.length === 0) return null;
    
    return (
      <HoverableDualAxisInfo 
        primaryAxisKeys={primaryAxisKeys} 
        finalSecondaryAxisKeys={finalSecondaryAxisKeys} 
      />
    );
  };

  // Helper function to render chart
  const renderChart = () => {
    if (!table || !table.chartData || table.chartData.length === 0) {
      return <Typography>No data available for visualization</Typography>;
    }

    const effectiveChartType = chartConfig?.chartType && chartConfig.chartType !== 'auto' 
      ? chartConfig.chartType 
      : (selectedChartType === 'auto' 
          ? getChartTypeForData(table.headers, table.rows, plot)
          : selectedChartType);

    let rawData = table.chartData;
    const xAxisKey = chartConfig?.xAxis || table.headers[0];
    let yAxisKeys = chartConfig?.yAxis?.length > 0 
      ? chartConfig.yAxis 
      : table.headers.slice(1);
    
    // Validate that yAxisKeys actually exist in the data
    if (rawData && rawData.length > 0) {
      const availableFields = Object.keys(rawData[0]);
      yAxisKeys = yAxisKeys.filter(key => availableFields.includes(key));
      
      // If no valid yAxisKeys found, use the first numeric field from data
      if (yAxisKeys.length === 0) {
        const numericFields = availableFields.filter(field => 
          typeof rawData[0][field] === 'number'
        );
        yAxisKeys = numericFields.length > 0 ? numericFields : ['Value'];
      }
    }
    
    const groupBy = chartConfig?.groupBy;

    // If groupBy is provided (e.g., StateName) and there is only one measure in yAxis
    // pivot the data to a wide format so each group becomes its own series/column
    if (groupBy && yAxisKeys.length === 1 && rawData && rawData.length > 0) {
      const valueKey = yAxisKeys[0];
      const xBuckets = new Map(); // x -> { x, group1: val, group2: val }
      const groups = new Set();
      for (const row of rawData) {
        const x = row[xAxisKey];
        const g = row[groupBy];
        const v = row[valueKey];
        if (!xBuckets.has(x)) {
          xBuckets.set(x, { [xAxisKey]: x });
        }
        const bucket = xBuckets.get(x);
        bucket[g] = v;
        groups.add(g);
      }
      rawData = Array.from(xBuckets.values());
      yAxisKeys = Array.from(groups.values());
    }

    // Enhanced secondary axis detection
    const secondaryAxisKeys = chartConfig?.yAxisSecondary || [];
    const singleSecondaryAxis = chartConfig?.secondaryAxis || '';
    
    if (singleSecondaryAxis && !secondaryAxisKeys.includes(singleSecondaryAxis)) {
      secondaryAxisKeys.push(singleSecondaryAxis);
    }
    
    // Robustly determine primary and secondary Y-axis columns
    let primaryAxisKeys = [];
    let finalSecondaryAxisKeys = [];
    
    if (groupBy && yAxisKeys.length > 0) {
      // For grouped data, distribute series between primary and secondary
      const allSeries = yAxisKeys;
      
      if (secondaryAxisKeys.length > 0) {
        primaryAxisKeys = allSeries.filter(series => !secondaryAxisKeys.includes(series));
        finalSecondaryAxisKeys = allSeries.filter(series => secondaryAxisKeys.includes(series));
      } else {
        const secondaryKeywords = [
          'growth', 'percentage', 'ratio', 'rate', 'change', 'increase', 'decrease'
        ];
        
        primaryAxisKeys = allSeries.filter(series => {
          if (!series || typeof series !== 'string') return false;
          const seriesLower = series.toLowerCase();
          return !secondaryKeywords.some(keyword => seriesLower.includes(keyword));
        });
        
        finalSecondaryAxisKeys = allSeries.filter(series => {
          if (!series || typeof series !== 'string') return false;
          const seriesLower = series.toLowerCase();
          return secondaryKeywords.some(keyword => seriesLower.includes(keyword));
        });
        
        // Additional check for specific column names
        const specificSecondaryColumns = ['GrowthPercentage'];
        
        const specificSecondary = allSeries.filter(series => 
          series && typeof series === 'string' && specificSecondaryColumns.some(col => series.includes(col))
        );
        
        if (specificSecondary.length > 0) {
          finalSecondaryAxisKeys = specificSecondary;
          primaryAxisKeys = allSeries.filter(series => 
            series && typeof series === 'string' && !specificSecondaryColumns.some(col => series.includes(col))
          );
        }
      }
    } else {
      // For non-grouped data
      primaryAxisKeys = yAxisKeys.filter(key => !secondaryAxisKeys.includes(key));
      finalSecondaryAxisKeys = secondaryAxisKeys;
      
      if (finalSecondaryAxisKeys.length === 0 && yAxisKeys.length > 1) {
        const secondaryKeywords = [
          'growth', 'percentage', 'ratio', 'rate', 'change', 'increase', 'decrease'
        ];
        
        const potentialSecondary = yAxisKeys.filter(key => {
          if (!key || typeof key !== 'string') return false;
          const keyLower = key.toLowerCase();
          return secondaryKeywords.some(keyword => keyLower.includes(keyword));
        });
        
        if (potentialSecondary.length > 0) {
          finalSecondaryAxisKeys = potentialSecondary;
          primaryAxisKeys = yAxisKeys.filter(key => !potentialSecondary.includes(key));
        }
      }
    }

    // Ensure we have at least some primary axis keys
    if (primaryAxisKeys.length === 0 && yAxisKeys.length > 0) {
      primaryAxisKeys = yAxisKeys.slice(0, Math.ceil(yAxisKeys.length / 2));
      finalSecondaryAxisKeys = yAxisKeys.slice(Math.ceil(yAxisKeys.length / 2));
    }

    // Filter data to only include selected columns
    const filteredData = rawData.map(item => {
      const filteredItem = { [xAxisKey]: item[xAxisKey] };
      [...primaryAxisKeys, ...finalSecondaryAxisKeys].forEach(key => {
        if (item[key] !== undefined) {
          filteredItem[key] = item[key];
        }
      });
      return filteredItem;
    });

    // Helper function to get axis label
    const getAxisLabel = (axisKeys, isSecondary = false) => {
      if (axisKeys.length === 0) return '';
      if (axisKeys.length === 1) return chartConfig?.valueLabel || axisKeys[0];
      return isSecondary ? 'Secondary Values' : (chartConfig?.valueLabel || 'Values');
    };

    switch (effectiveChartType) {
      case 'line':
        return (
          <Box sx={{ position: 'relative' }}>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={filteredData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={xAxisKey} />
                <YAxis 
                  yAxisId="left" 
                  label={{ value: getAxisLabel(primaryAxisKeys, false), angle: -90, position: 'insideLeft' }}
                />
                {finalSecondaryAxisKeys.length > 0 && (
                  <YAxis 
                    yAxisId="right" 
                    orientation="right"
                    label={{ value: getAxisLabel(finalSecondaryAxisKeys, true), angle: 90, position: 'insideRight' }}
                  />
                )}
                <RechartsTooltip content={({ active, payload, label }) => {
                  if (!active || !payload) return null;
                  const items = [...payload].sort((a,b) => {
                    const av = typeof a.value === 'number' ? a.value : parseFloat(a.value);
                    const bv = typeof b.value === 'number' ? b.value : parseFloat(b.value);
                    return (isNaN(bv)?0:bv) - (isNaN(av)?0:av);
                  });
                  return (
                    <Paper sx={{ p: 1 }}>
                      <Typography variant="caption" sx={{ fontWeight: 'bold' }}>{label}</Typography>
                      {items.map((it, idx) => (
                        <Box key={idx} sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                          <span style={{ width: 10, height: 10, background: it.color }} />
                          <Typography variant="caption">{it.name}: {it.value}</Typography>
                        </Box>
                      ))}
                    </Paper>
                  );
                }} />
                <Legend />
                
                {primaryAxisKeys.map((key, index) => (
                  <Line 
                    key={key} 
                    type="monotone" 
                    dataKey={key} 
                    stroke={`hsl(${index * 60}, 70%, 50%)`}
                    yAxisId="left"
                    strokeWidth={2}
                  />
                ))}
                
                {finalSecondaryAxisKeys.map((key, index) => (
                  <Line 
                    key={key} 
                    type="monotone" 
                    dataKey={key} 
                    stroke={`hsl(${(index + primaryAxisKeys.length) * 60}, 70%, 50%)`}
                    yAxisId="right"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </Box>
        );

      case 'bar':
        return (
          <Box sx={{ position: 'relative' }}>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={filteredData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={xAxisKey} />
                <YAxis 
                  yAxisId="left" 
                  label={{ value: getAxisLabel(primaryAxisKeys, false), angle: -90, position: 'insideLeft' }}
                />
                {finalSecondaryAxisKeys.length > 0 && (
                  <YAxis 
                    yAxisId="right" 
                    orientation="right"
                    label={{ value: getAxisLabel(finalSecondaryAxisKeys, true), angle: 90, position: 'insideRight' }}
                  />
                )}
                <RechartsTooltip content={({ active, payload, label }) => {
                  if (!active || !payload) return null;
                  const items = [...payload].sort((a,b) => {
                    const av = typeof a.value === 'number' ? a.value : parseFloat(a.value);
                    const bv = typeof b.value === 'number' ? b.value : parseFloat(b.value);
                    return (isNaN(bv)?0:bv) - (isNaN(av)?0:av);
                  });
                  return (
                    <Paper sx={{ p: 1 }}>
                      <Typography variant="caption" sx={{ fontWeight: 'bold' }}>{label}</Typography>
                      {items.map((it, idx) => (
                        <Box key={idx} sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                          <span style={{ width: 10, height: 10, background: it.color }} />
                          <Typography variant="caption">{it.name}: {it.value}</Typography>
                        </Box>
                      ))}
                    </Paper>
                  );
                }} />
                <Legend />
                
                {primaryAxisKeys.map((key, index) => (
                  <Bar 
                    key={key} 
                    dataKey={key} 
                    fill={`hsl(${index * 60}, 70%, 50%)`}
                    yAxisId="left"
                  />
                ))}
                
                {finalSecondaryAxisKeys.map((key, index) => (
                  <Bar 
                    key={key} 
                    dataKey={key} 
                    fill={`hsl(${(index + primaryAxisKeys.length) * 60}, 70%, 50%)`}
                    yAxisId="right"
                    opacity={0.7}
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </Box>
        );

      case 'dualAxisBarLine':
        return (
          <Box sx={{ position: 'relative' }}>
            <ResponsiveContainer width="100%" height={400}>
              <ComposedChart data={filteredData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={xAxisKey} />
                <YAxis 
                  yAxisId="left" 
                  label={{ value: getAxisLabel(primaryAxisKeys, false), angle: -90, position: 'insideLeft' }}
                />
                {finalSecondaryAxisKeys.length > 0 && (
                  <YAxis 
                    yAxisId="right" 
                    orientation="right"
                    label={{ value: getAxisLabel(finalSecondaryAxisKeys, true), angle: 90, position: 'insideRight' }}
                  />
                )}
                <RechartsTooltip content={({ active, payload, label }) => {
                  if (!active || !payload) return null;
                  const items = [...payload].sort((a,b) => {
                    const av = typeof a.value === 'number' ? a.value : parseFloat(a.value);
                    const bv = typeof b.value === 'number' ? b.value : parseFloat(b.value);
                    return (isNaN(bv)?0:bv) - (isNaN(av)?0:av);
                  });
                  return (
                    <Paper sx={{ p: 1 }}>
                      <Typography variant="caption" sx={{ fontWeight: 'bold' }}>{label}</Typography>
                      {items.map((it, idx) => (
                        <Box key={idx} sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                          <span style={{ width: 10, height: 10, background: it.color }} />
                          <Typography variant="caption">{it.name}: {it.value}</Typography>
                        </Box>
                      ))}
                    </Paper>
                  );
                }} />
                <Legend />
                
                {primaryAxisKeys.map((key, index) => (
                  <Bar 
                    key={key} 
                    dataKey={key} 
                    fill={`hsl(${index * 60}, 70%, 50%)`}
                    yAxisId="left"
                  />
                ))}
                
                {finalSecondaryAxisKeys.map((key, index) => (
                  <Line 
                    key={key} 
                    type="monotone" 
                    dataKey={key} 
                    stroke={`hsl(${(index + primaryAxisKeys.length) * 60}, 70%, 50%)`}
                    yAxisId="right"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                  />
                ))}
              </ComposedChart>
            </ResponsiveContainer>
          </Box>
        );

      case 'dualAxisLine':
        return (
          <Box sx={{ position: 'relative' }}>
            <ResponsiveContainer width="100%" height={400}>
              <ComposedChart data={filteredData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={xAxisKey} />
                <YAxis 
                  yAxisId="left" 
                  label={{ value: getAxisLabel(primaryAxisKeys, false), angle: -90, position: 'insideLeft' }}
                />
                {finalSecondaryAxisKeys.length > 0 && (
                  <YAxis 
                    yAxisId="right" 
                    orientation="right"
                    label={{ value: getAxisLabel(finalSecondaryAxisKeys, true), angle: 90, position: 'insideRight' }}
                  />
                )}
                <RechartsTooltip content={({ active, payload, label }) => {
                  if (!active || !payload) return null;
                  const items = [...payload].sort((a,b) => {
                    const av = typeof a.value === 'number' ? a.value : parseFloat(a.value);
                    const bv = typeof b.value === 'number' ? b.value : parseFloat(b.value);
                    return (isNaN(bv)?0:bv) - (isNaN(av)?0:av);
                  });
                  return (
                    <Paper sx={{ p: 1 }}>
                      <Typography variant="caption" sx={{ fontWeight: 'bold' }}>{label}</Typography>
                      {items.map((it, idx) => (
                        <Box key={idx} sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                          <span style={{ width: 10, height: 10, background: it.color }} />
                          <Typography variant="caption">{it.name}: {it.value}</Typography>
                        </Box>
                      ))}
                    </Paper>
                  );
                }} />
                <Legend />
                
                {primaryAxisKeys.map((key, index) => (
                  <Line 
                    key={key} 
                    type="monotone" 
                    dataKey={key} 
                    stroke={`hsl(${index * 60}, 70%, 50%)`}
                    yAxisId="left"
                    strokeWidth={2}
                  />
                ))}
                
                {finalSecondaryAxisKeys.map((key, index) => (
                  <Line 
                    key={key} 
                    type="monotone" 
                    dataKey={key} 
                    stroke={`hsl(${(index + primaryAxisKeys.length) * 60}, 70%, 50%)`}
                    yAxisId="right"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                  />
                ))}
              </ComposedChart>
            </ResponsiveContainer>
          </Box>
        );

      case 'pie':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <PieChart>
              <Pie
                data={filteredData}
                dataKey={yAxisKeys[0]}
                nameKey={xAxisKey}
                cx="50%"
                cy="50%"
                outerRadius={150}
                fill="#8884d8"
                label
              />
              <RechartsTooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        );

      default:
        return <Typography>Chart type not supported</Typography>;
    }
  };

  // Helper function to render table
  const renderTable = () => {
    if (!table || !table.rows || table.rows.length === 0) {
      return <Typography>No data available</Typography>;
    }

    const columns = table.headers.map((header, index) => ({
      field: header,
      headerName: header,
      width: 150,
      flex: 1,
    }));

    const rows = table.rows.map((row, index) => {
      if (Array.isArray(row)) {
        // Convert array format to object format
        const rowObj = { id: index };
        table.headers.forEach((header, headerIndex) => {
          rowObj[header] = row[headerIndex];
        });
        return rowObj;
      } else {
        // Handle object format
        return {
          id: index,
          ...row,
        };
      }
    });

    return (
      <Box sx={{ height: 400, width: '100%' }}>
        <DataGrid
          rows={rows}
          columns={columns}
          pageSize={10}
          rowsPerPageOptions={[5, 10, 25, 50]}
          disableSelectionOnClick
          sx={{
            '& .MuiDataGrid-root': {
              border: 'none',
            },
            '& .MuiDataGrid-cell': {
              borderBottom: '1px solid #e0e0e0',
            },
            '& .MuiDataGrid-columnHeaders': {
              backgroundColor: '#f5f5f5',
              borderBottom: '2px solid #e0e0e0',
            },
            '& .MuiDataGrid-row:hover': {
              backgroundColor: '#f8f9fa',
            },
          }}
        />
      </Box>
    );
  };

  // Only return null if there's absolutely no data to display
  if (!table || ((!table.chartData || table.chartData.length === 0) && (!table.rows || table.rows.length === 0))) {
    return null;
  }

  return (
    <Paper style={{ padding: 16, marginBottom: 24 }}>
      <Box sx={{ borderBottom: 1, borderColor: 'divider', marginBottom: 2 }}>
        <Tabs value={viewMode} onChange={(e, newValue) => onViewModeChange(newValue)}>
          <Tab label="Chart View" value="chart" />
          <Tab label="Table View" value="table" />
        </Tabs>
      </Box>
      
      {viewMode === "chart" && (
        <Box>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Typography variant="h6" gutterBottom>
              {getChartTitle()}
            </Typography>
            
            {/* Hoverable Dual-Axis Info */}
            {axisInfo && renderHoverableDualAxisInfo(axisInfo.primaryAxisKeys, axisInfo.finalSecondaryAxisKeys)}
          </Box>
          
          {/* Chart Type Selector and Chart Builder */}
          <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
            <Typography variant="body2" color="text.secondary">Chart Type:</Typography>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <Select
                value={selectedChartType}
                onChange={(e) => onChartTypeChange(e.target.value)}
                displayEmpty
              >
                <MenuItem value="auto">Auto (AI Recommended)</MenuItem>
                <MenuItem value="dualAxisBarLine">Dual-Axis Bar + Line</MenuItem>
                <MenuItem value="dualAxisLine">Dual-Axis Line</MenuItem>
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
              onClick={onToggleChartBuilder}
              startIcon={<DragIndicatorIcon />}
            >
              🎨 Chart Builder
            </Button>
            
            {/* AI Visualization Info */}
            {plot?.options?.dataType && (
              <Chip 
                label={`AI: ${plot.options.dataType}`}
                size="small" 
                color="info" 
                variant="outlined"
              />
            )}
          </Box>
          
          {/* Chart Rendering */}
          {renderChart()}
        </Box>
      )}
      
      {viewMode === "table" && (
        <Box>
          <Typography variant="h6" gutterBottom>Data Table</Typography>
          {renderTable()}
        </Box>
      )}
      
      {/* Chart Builder Dialog */}
      <ChartBuilder
        open={showChartBuilder}
        onClose={onToggleChartBuilder}
        onApply={onUpdateChartConfig}
        availableColumns={table?.headers || []}
        currentConfig={chartConfig}
      />
    </Paper>
  );
};

export default DataVisualization; 