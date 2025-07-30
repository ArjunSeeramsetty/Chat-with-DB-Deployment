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

  // Helper function to get chart type for data
  const getChartTypeForData = (headers, rows, plotRecommendation) => {
    if (!headers || !rows || rows.length === 0) return 'bar';
    
    const numColumns = headers.length;
    const numRows = rows.length;
    
    // Check if it's time series data - check all headers, not just the first one
    const isTimeSeries = headers.some(header => 
      header.toLowerCase().includes('date') || 
      header.toLowerCase().includes('month') || 
      header.toLowerCase().includes('year') ||
      header.toLowerCase().includes('quarter')
    );
    
    // Check if it's growth data
    const isGrowthData = plotRecommendation?.options?.dataType === "growth_time_series";
    
    // Check if it's monthly multi-state data
    const hasMonthColumn = headers.some(header => header.toLowerCase().includes('month'));
    const hasStateColumn = headers.some(header => header.toLowerCase().includes('state'));
    const isMonthlyMultiState = hasMonthColumn && hasStateColumn;
    
    // Check for dual-axis potential
    const hasGrowthColumns = headers.some(header => 
      header.toLowerCase().includes('growth') || 
      header.toLowerCase().includes('percentage') ||
      header.toLowerCase().includes('ratio') ||
      header.toLowerCase().includes('rate')
    );
    const hasTotalColumns = headers.some(header => 
      !header.toLowerCase().includes('growth') && 
      !header.toLowerCase().includes('percentage') &&
      !header.toLowerCase().includes('ratio') &&
      !header.toLowerCase().includes('rate') &&
      (header.toLowerCase().includes('total') || 
       header.toLowerCase().includes('value') ||
       header.toLowerCase().includes('amount') ||
       header.toLowerCase().includes('generation'))
    );
    const hasDualAxisPotential = hasGrowthColumns && hasTotalColumns;
    
    // Prioritize dual-axis charts when appropriate
    if (hasDualAxisPotential && isTimeSeries) {
      return 'dualAxisLine';
    } else if (hasDualAxisPotential) {
      return 'dualAxisBarLine';
    } else if (isGrowthData) {
      return 'multiLine';
    } else if (isMonthlyMultiState) {
      return 'multiLine';
    } else if (isTimeSeries && numRows > 1) {
      return 'line';
    } else if (numColumns === 2 && numRows <= 10) {
      return 'pie';
    } else if (numRows > 50) {
      return 'line';
    }
    return 'bar';
  };

  // Helper function to process chart data for time series
  const processTimeSeriesData = (data, xAxisKey) => {
    if (!data || data.length === 0) return data;
    
    // Check if it's time series data - check the xAxisKey and also look for month columns
    const isTimeSeries = xAxisKey?.toLowerCase().includes('month') || 
                        xAxisKey?.toLowerCase().includes('date') || 
                        xAxisKey?.toLowerCase().includes('year') ||
                        // Also check if any column in the data contains month information
                        data.some(item => {
                          const keys = Object.keys(item);
                          return keys.some(key => key.toLowerCase().includes('month'));
                        });
    
    if (!isTimeSeries) return data;
    
    // Sort by x-axis values for time series
    return [...data].sort((a, b) => {
      const aVal = a[xAxisKey];
      const bVal = b[xAxisKey];
      
      // Handle numeric values
      if (!isNaN(aVal) && !isNaN(bVal)) {
        return Number(aVal) - Number(bVal);
      }
      
      // Handle month names
      const months = ['january', 'february', 'march', 'april', 'may', 'june', 
                     'july', 'august', 'september', 'october', 'november', 'december'];
      const aMonth = months.indexOf(aVal?.toLowerCase());
      const bMonth = months.indexOf(bVal?.toLowerCase());
      
      if (aMonth !== -1 && bMonth !== -1) {
        return aMonth - bMonth;
      }
      
      // Default string comparison
      return String(aVal).localeCompare(String(bVal));
    });
  };

  // Helper function to determine if data is categorical
  const isCategoricalData = (data, xAxisKey) => {
    if (!data || data.length === 0) return false;
    
    // Check if x-axis values are strings and likely categorical
    const uniqueValues = [...new Set(data.map(item => item[xAxisKey]))];
    const isStringData = data.every(item => typeof item[xAxisKey] === 'string');
    const hasManyCategories = uniqueValues.length > 5;
    
    return isStringData && hasManyCategories;
  };

  // Helper function to get chart title with dual-axis indicator
  const getChartTitle = () => {
    if (!table || !table.headers || table.headers.length === 0) {
      return 'Chart';
    }
    
    const xAxisKey = chartConfig?.xAxis || table.headers[0];
    const baseTitle = chartConfig?.title || `Chart: ${xAxisKey}`;
    
    // Check if we have secondary axis data to determine if it's dual-axis
    const secondaryAxisKeys = chartConfig?.yAxisSecondary || [];
    const singleSecondaryAxis = chartConfig?.secondaryAxis || '';
    const hasSecondaryAxis = secondaryAxisKeys.length > 0 || singleSecondaryAxis;
    
    if (hasSecondaryAxis) {
      return `${baseTitle} (Dual-Axis)`;
    }
    return baseTitle;
  };

  // Helper function to render dual-axis indicator
  const renderDualAxisIndicator = (primaryAxisKeys, finalSecondaryAxisKeys) => {
    if (finalSecondaryAxisKeys.length === 0) return null;
    
    return (
      <Box sx={{ 
        position: 'absolute', 
        top: 10, 
        right: 10, 
        backgroundColor: 'rgba(255, 255, 255, 0.9)', 
        padding: 1, 
        borderRadius: 1,
        border: '1px solid #ccc',
        zIndex: 1
      }}>
        <Typography variant="caption" color="text.secondary">
          Dual-Axis Chart
        </Typography>
        <Box sx={{ mt: 0.5 }}>
          <Typography variant="caption" color="primary">
            Left: {primaryAxisKeys.join(', ')}
          </Typography>
          <br />
          <Typography variant="caption" color="secondary">
            Right: {finalSecondaryAxisKeys.join(', ')}
          </Typography>
        </Box>
      </Box>
    );
  };

  // Helper function to process data based on Group By configuration
  const processGroupByData = (data, groupBy, xAxisKey, yAxisKeys) => {
    if (!groupBy || !data || data.length === 0) return data;
    
    // Get unique values for the group by column
    const uniqueGroupValues = [...new Set(data.map(item => item[groupBy]))].sort();
    
    // Create separate series for each group value
    const seriesData = {};
    
    uniqueGroupValues.forEach(groupValue => {
      // Filter data for this group value
      const groupData = data.filter(item => item[groupBy] === groupValue);
      
      // Create series data for this group
      seriesData[groupValue] = groupData.map(item => {
        const seriesItem = { [xAxisKey]: item[xAxisKey] };
        yAxisKeys.forEach(yKey => {
          if (item[yKey] !== undefined) {
            seriesItem[yKey] = item[yKey];
          }
        });
        return seriesItem;
      });
    });
    
    // For multi-line charts, we need to restructure the data
    // Each group becomes a separate dataKey
    const result = [];
    
    // Get all unique x-axis values and sort them properly
    let allXValues = [...new Set(data.map(item => item[xAxisKey]))];
    
    // Sort X values properly (handle numeric vs string sorting)
    if (allXValues.every(val => !isNaN(Number(val)))) {
      // Numeric sorting
      allXValues = allXValues.sort((a, b) => Number(a) - Number(b));
    } else {
      // String sorting
      allXValues = allXValues.sort();
    }
    
    allXValues.forEach(xValue => {
      const resultItem = { [xAxisKey]: xValue };
      
      uniqueGroupValues.forEach(groupValue => {
        const groupItem = seriesData[groupValue]?.find(item => item[xAxisKey] === xValue);
        if (groupItem) {
          yAxisKeys.forEach(yKey => {
            if (groupItem[yKey] !== undefined) {
              resultItem[`${groupValue}_${yKey}`] = groupItem[yKey];
            }
          });
        }
      });
      
      result.push(resultItem);
    });
    
    return {
      data: result,
      seriesKeys: uniqueGroupValues.map(groupValue => 
        yAxisKeys.map(yKey => `${groupValue}_${yKey}`)
      ).flat()
    };
  };

  // Helper function to render chart
  const renderChart = () => {
    if (!table || !table.chartData || table.chartData.length === 0) {
      return <Typography>No data available for visualization</Typography>;
    }

    // Use chart config if available, otherwise use auto-detection
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
    const groupBy = chartConfig?.groupBy;
    
    // Process data for time series
    rawData = processTimeSeriesData(rawData, xAxisKey);
    
    // Apply Group By processing if specified
    let seriesKeys = [];
    if (groupBy) {
      const groupByResult = processGroupByData(rawData, groupBy, xAxisKey, yAxisKeys);
      rawData = groupByResult.data;
      seriesKeys = groupByResult.seriesKeys;
      
      // Update yAxisKeys to include the grouped series
      if (seriesKeys.length > 0) {
        yAxisKeys = seriesKeys;
      }
    }
    
    // Check if data is categorical
    const isCategorical = isCategoricalData(rawData, xAxisKey);

    // Enhanced secondary axis detection
    const secondaryAxisKeys = chartConfig?.yAxisSecondary || [];
    const singleSecondaryAxis = chartConfig?.secondaryAxis || '';
    
    // If a single secondary axis is specified, use it
    if (singleSecondaryAxis && !secondaryAxisKeys.includes(singleSecondaryAxis)) {
      secondaryAxisKeys.push(singleSecondaryAxis);
    }
    
    // Auto-detect secondary axis for growth time series data
    if (plot?.options?.dataType === 'growth_time_series' && plot?.options?.growthFieldName) {
      const growthField = plot.options.growthFieldName;
      if (!secondaryAxisKeys.includes(growthField)) {
        secondaryAxisKeys.push(growthField);
      }
    }
    
    // Robustly determine primary and secondary Y-axis columns
    let primaryAxisKeys = [];
    let finalSecondaryAxisKeys = [];
    
    if (groupBy && seriesKeys.length > 0) {
      // For grouped data, distribute series between primary and secondary
      const allSeries = seriesKeys;
      
      if (secondaryAxisKeys.length > 0) {
        // If secondary axis is specified, use it to filter series
        primaryAxisKeys = allSeries.filter(series => !secondaryAxisKeys.includes(series));
        finalSecondaryAxisKeys = allSeries.filter(series => secondaryAxisKeys.includes(series));
      } else {
        // Enhanced auto-distribution: put growth/percentage series on secondary
        const secondaryKeywords = [
          'growth', 'percentage', 'ratio', 'rate', 'change', 'increase', 'decrease',
          'growthpercentage', 'growth_percentage', 'growth_rate', 'change_rate',
          'percent', 'pct', 'ratio', 'proportion', 'share'
        ];
        
        primaryAxisKeys = allSeries.filter(series => {
          const seriesLower = series.toLowerCase();
          return !secondaryKeywords.some(keyword => seriesLower.includes(keyword));
        });
        
        finalSecondaryAxisKeys = allSeries.filter(series => {
          const seriesLower = series.toLowerCase();
          return secondaryKeywords.some(keyword => seriesLower.includes(keyword));
        });
        
        // If no secondary candidates, put half on secondary
        if (finalSecondaryAxisKeys.length === 0 && allSeries.length > 1) {
          const halfIndex = Math.ceil(allSeries.length / 2);
          primaryAxisKeys = allSeries.slice(0, halfIndex);
          finalSecondaryAxisKeys = allSeries.slice(halfIndex);
        }
      }
    } else {
      // For non-grouped data, use the original logic
      primaryAxisKeys = yAxisKeys.filter(key => !secondaryAxisKeys.includes(key));
      finalSecondaryAxisKeys = secondaryAxisKeys;
      
      // Auto-detect secondary axis for non-grouped data
      if (finalSecondaryAxisKeys.length === 0 && yAxisKeys.length > 1) {
        const secondaryKeywords = [
          'growth', 'percentage', 'ratio', 'rate', 'change', 'increase', 'decrease',
          'growthpercentage', 'growth_percentage', 'growth_rate', 'change_rate',
          'percent', 'pct', 'ratio', 'proportion', 'share'
        ];
        
        const potentialSecondary = yAxisKeys.filter(key => {
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

    // If axes are not explicitly defined in plot options, intelligently find them
    if (primaryAxisKeys.length === 0 || (finalSecondaryAxisKeys.length === 0 && yAxisKeys.length > 1)) {
      const numericalCols = yAxisKeys.filter(key => {
        // Check if all values in this column are numbers
        return filteredData.every(row => {
          const value = row[key];
          return typeof value === 'number' && !isNaN(value);
        });
      });
      
      const growthCol = numericalCols.find(col => 
        col.toLowerCase().includes('growth') || 
        col.toLowerCase().includes('percent') ||
        col.toLowerCase().includes('ratio') ||
        col.toLowerCase().includes('rate')
      );
      
      const totalCol = numericalCols.find(col => 
        !col.toLowerCase().includes('growth') && 
        !col.toLowerCase().includes('percent') &&
        !col.toLowerCase().includes('ratio') &&
        !col.toLowerCase().includes('rate')
      );

      if (primaryAxisKeys.length === 0) {
        primaryAxisKeys = totalCol ? [totalCol] : (numericalCols.length > 0 ? [numericalCols[0]] : []);
      }
      
      if (finalSecondaryAxisKeys.length === 0 && numericalCols.length > 1) {
        finalSecondaryAxisKeys = growthCol ? [growthCol] : [numericalCols[1]];
      }
    }
    
    console.log('Dual-axis chart config:', {
      xAxis: xAxisKey,
      primaryAxisKeys,
      finalSecondaryAxisKeys,
      plotOptions: plot?.options
    });

    // Ensure both axes are valid before rendering
    if (primaryAxisKeys.length === 0) {
      return (
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
          <Typography color="error">Chart requires at least one numerical column. Please check the data.</Typography>
        </Box>
      );
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

    console.log('Chart Debug Info:', {
      effectiveChartType,
      xAxisKey,
      primaryAxisKeys,
      finalSecondaryAxisKeys,
      groupBy,
      seriesKeys,
      filteredDataLength: filteredData.length,
      sampleData: filteredData.slice(0, 2)
    });

    // Helper function to get axis label
    const getAxisLabel = (axisKeys, isSecondary = false) => {
      if (axisKeys.length === 0) return '';
      if (axisKeys.length === 1) return axisKeys[0];
      
      // For multiple keys, create a descriptive label
      const commonPrefix = axisKeys[0].split('_')[0];
      const allSamePrefix = axisKeys.every(key => key.startsWith(commonPrefix));
      
      if (allSamePrefix && commonPrefix.length > 2) {
        return `${commonPrefix} (${isSecondary ? 'Secondary' : 'Primary'})`;
      }
      
      return isSecondary ? 'Secondary Values' : 'Primary Values';
    };

    switch (effectiveChartType) {
      case 'line':
        return (
          <Box sx={{ position: 'relative' }}>
            {renderDualAxisIndicator(primaryAxisKeys, finalSecondaryAxisKeys)}
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={filteredData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey={xAxisKey} 
                  angle={isCategorical ? -45 : 0}
                  textAnchor={isCategorical ? "end" : "middle"}
                  height={isCategorical ? 100 : 60}
                  interval={isCategorical ? Math.ceil(filteredData.length / 10) : 0}
                  tick={{ fontSize: isCategorical ? 10 : 12 }}
                />
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
                <RechartsTooltip />
                {chartConfig?.showLegend !== false && <Legend />}
                
                {/* Render primary axis lines */}
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
                
                {/* Render secondary axis lines */}
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
            {renderDualAxisIndicator(primaryAxisKeys, finalSecondaryAxisKeys)}
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={filteredData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey={xAxisKey} 
                  angle={isCategorical ? -45 : 0}
                  textAnchor={isCategorical ? "end" : "middle"}
                  height={isCategorical ? 100 : 60}
                  interval={isCategorical ? Math.ceil(filteredData.length / 10) : 0}
                  tick={{ fontSize: isCategorical ? 10 : 12 }}
                />
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
                <RechartsTooltip />
                {chartConfig?.showLegend !== false && <Legend />}
                
                {/* Render primary axis bars */}
                {primaryAxisKeys.map((key, index) => (
                  <Bar 
                    key={key} 
                    dataKey={key} 
                    fill={`hsl(${index * 60}, 70%, 50%)`}
                    yAxisId="left"
                  />
                ))}
                
                {/* Render secondary axis bars */}
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

      case 'multiLine':
        return (
          <Box sx={{ position: 'relative' }}>
            {renderDualAxisIndicator(primaryAxisKeys, finalSecondaryAxisKeys)}
            <ResponsiveContainer width="100%" height={400}>
              <ComposedChart data={filteredData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey={xAxisKey} 
                  angle={isCategorical ? -45 : 0}
                  textAnchor={isCategorical ? "end" : "middle"}
                  height={isCategorical ? 100 : 60}
                  interval={isCategorical ? Math.ceil(filteredData.length / 10) : 0}
                  tick={{ fontSize: isCategorical ? 10 : 12 }}
                />
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
                <RechartsTooltip />
                {chartConfig?.showLegend !== false && <Legend />}
                
                {/* Render primary axis lines */}
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
                
                {/* Render secondary axis lines */}
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

      case 'dualAxisBarLine':
        return (
          <Box sx={{ position: 'relative' }}>
            {renderDualAxisIndicator(primaryAxisKeys, finalSecondaryAxisKeys)}
            <ResponsiveContainer width="100%" height={400}>
              <ComposedChart data={filteredData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey={xAxisKey} 
                  angle={isCategorical ? -45 : 0}
                  textAnchor={isCategorical ? "end" : "middle"}
                  height={isCategorical ? 100 : 60}
                  interval={isCategorical ? Math.ceil(filteredData.length / 10) : 0}
                  tick={{ fontSize: isCategorical ? 10 : 12 }}
                />
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
                <RechartsTooltip />
                {chartConfig?.showLegend !== false && <Legend />}
                
                {/* Render primary axis bars */}
                {primaryAxisKeys.map((key, index) => (
                  <Bar 
                    key={key} 
                    dataKey={key} 
                    fill={`hsl(${index * 60}, 70%, 50%)`}
                    yAxisId="left"
                  />
                ))}
                
                {/* Render secondary axis lines */}
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
            {renderDualAxisIndicator(primaryAxisKeys, finalSecondaryAxisKeys)}
            <ResponsiveContainer width="100%" height={400}>
              <ComposedChart data={filteredData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey={xAxisKey} 
                  angle={isCategorical ? -45 : 0}
                  textAnchor={isCategorical ? "end" : "middle"}
                  height={isCategorical ? 100 : 60}
                  interval={isCategorical ? Math.ceil(filteredData.length / 10) : 0}
                  tick={{ fontSize: isCategorical ? 10 : 12 }}
                />
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
                <RechartsTooltip />
                {chartConfig?.showLegend !== false && <Legend />}
                
                {/* Render primary axis lines */}
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
                
                {/* Render secondary axis lines */}
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

      case 'stackedBar':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={filteredData} stackOffset="expand">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey={xAxisKey} 
                angle={isCategorical ? -45 : 0}
                textAnchor={isCategorical ? "end" : "middle"}
                height={isCategorical ? 100 : 60}
                interval={isCategorical ? Math.ceil(filteredData.length / 10) : 0}
                tick={{ fontSize: isCategorical ? 10 : 12 }}
              />
              <YAxis />
              <RechartsTooltip />
              {chartConfig?.showLegend !== false && <Legend />}
              {yAxisKeys.map((key, index) => (
                <Bar key={key} dataKey={key} stackId="a" fill={`hsl(${index * 60}, 70%, 50%)`} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        );

      case 'groupedBar':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={filteredData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey={xAxisKey} 
                angle={isCategorical ? -45 : 0}
                textAnchor={isCategorical ? "end" : "middle"}
                height={isCategorical ? 100 : 60}
                interval={isCategorical ? Math.ceil(filteredData.length / 10) : 0}
                tick={{ fontSize: isCategorical ? 10 : 12 }}
              />
              <YAxis />
              <RechartsTooltip />
              {chartConfig?.showLegend !== false && <Legend />}
              {yAxisKeys.map((key, index) => (
                <Bar key={key} dataKey={key} fill={`hsl(${index * 60}, 70%, 50%, 0.8)`} />
              ))}
            </BarChart>
          </ResponsiveContainer>
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

    const rows = table.rows.map((row, index) => ({
      id: index,
      ...row,
    }));

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

  if (!plot || !table || !table.chartData || table.chartData.length === 0) {
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
          <Typography variant="h6" gutterBottom>
            {getChartTitle()}
          </Typography>
          
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