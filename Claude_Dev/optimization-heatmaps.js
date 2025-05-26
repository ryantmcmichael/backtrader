// Extract React and ReactDOM from global scope
const { useState, useEffect, useMemo } = React;

// Select component remains the same...
const Select = ({ label, value, onChange, options }) => {
    return React.createElement('div', { className: 'mb-4' },
        React.createElement('label', { className: 'block text-sm font-medium mb-1' }, label),
        React.createElement('select', {
            value: value,
            onChange: (e) => onChange(e.target.value),
            className: 'w-full p-2 border rounded shadow-sm'
        }, options.map(option =>
            React.createElement('option', {
                key: option.value,
                value: option.value
            }, option.label)
        ))
    );
};

// HeatmapCell component remains the same...
const HeatmapCell = ({ value, count, min, max, isOptimal }) => {
    if (value === null) return React.createElement('div', {
        className: 'flex-1 aspect-square bg-gray-200'
    });

    const normalizedValue = (value - min) / (max - min);
    const backgroundColor = `rgb(${Math.floor(255 * (1 - normalizedValue))}, ${Math.floor(255 * normalizedValue)}, 0)`;

    return React.createElement('div', {
        className: 'flex-1 aspect-square relative group',
        style: {
            backgroundColor,
            border: isOptimal ? '2px solid blue' : 'none',
            boxShadow: isOptimal ? '0 0 10px rgba(0,0,255,0.5)' : 'none'
        }
    },
        React.createElement('div', {
            className: 'opacity-0 group-hover:opacity-100 absolute bottom-full left-1/2 transform -translate-x-1/2 bg-black text-white p-2 rounded text-sm whitespace-nowrap z-10'
        }, `Return: ${value.toFixed(1)}%, Sample size: ${count}${isOptimal ? ' (Best configuration)' : ''}`)
    );
};

// Returns Plot component
const ReturnsPlot = ({ data }) => {
    const canvasRef = React.useRef(null);

    useEffect(() => {
        if (!data || !canvasRef.current) return;

        const ctx = canvasRef.current.getContext('2d');
        if (window.myChart) window.myChart.destroy();

        const dates = Object.keys(data);
        const returns = Object.values(data);

        // Calculate cumulative returns
        let cumulative = 0;
        const cumulativeReturns = returns.map(r => (cumulative += r));

        window.myChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [{
                    label: 'Cumulative Return',
                    data: cumulativeReturns,
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        ticks: {
                            callback: function(value) {
                                return value.toFixed(1) + '%';
                            }
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Portfolio Returns'
                    }
                }
            }
        });
    }, [data]);

    return React.createElement('canvas', {
        ref: canvasRef,
        className: 'w-full h-96'
    });
};

// Main component
const OptimizationHeatmaps = () => {
    const [data, setData] = useState([]);
    const [filters, setFilters] = useState({
        use_mg_pos_signal: '',
        use_mg_40_signal: '',
        use_sp500_signal: '',
        use_sector_signals: '',
        entry_time_offset: ''
    });
    const [returnsData, setReturnsData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // Load data
    useEffect(() => {
        fetch('Optimization Results/detailed_results.csv')
            .then(response => response.text())
            .then(text => {
                const rows = text.split('\n').map(row => row.split(','));
                const headers = rows[0];

                // Parse the data first
                const parsedData = rows.slice(1)
                    .filter(row => row.length === headers.length)
                    .map(row => {
                        const obj = {};
                        headers.forEach((header, i) => {
                            const value = row[i]?.trim();
                            obj[header] = value === 'True' ? true :
                                         value === 'False' ? false :
                                         isNaN(value) ? value : Number(value);
                        });
                        return obj;
                    })
                    .filter(row => row.cumulative_return != null);

                // Sort by score (descending) and add runIndex
                const sortedData = [...parsedData]
                    .sort((a, b) => b.score - a.score)
                    .map((row, index) => ({
                        ...row,
                        runIndex: index + 1
                    }));

                setData(sortedData);
                setLoading(false);
            })
            .catch(err => {
                setError(err.message);
                setLoading(false);
            });
    }, []);

    // Get heatmap data with current filters
    const { cells, xValues, yValues, optimalConfig } = useMemo(() => {
        if (!data.length) return { cells: [], xValues: [], yValues: [], optimalConfig: null };

        const xValues = [...new Set(data.map(d => d.profit_target))].sort((a, b) => a - b);
        const yValues = [...new Set(data.map(d => d.loss_target))].sort((a, b) => a - b);

        let bestReturn = -Infinity;
        let optimalConfig = null;

        const cells = yValues.map((y, yi) =>
            xValues.map((x, xi) => {
                const matchingRows = data.filter(d =>
                    d.profit_target === x &&
                    d.loss_target === y &&
                    Object.entries(filters).every(([key, value]) =>
                        value === '' || d[key] === (key === 'entry_time_offset' ? Number(value) : value === 'true')
                    )
                );

                if (matchingRows.length) {
                    const avgReturn = matchingRows.reduce((sum, d) => sum + d.cumulative_return, 0) / matchingRows.length;
                    if (avgReturn > bestReturn) {
                        bestReturn = avgReturn;
                        // Find the row with the highest score among matching rows
                        const bestRow = matchingRows.reduce((best, current) =>
                            current.score > best.score ? current : best
                        );
                        optimalConfig = {
                            profit_target: x,
                            loss_target: y,
                            return: avgReturn,
                            runIndex: bestRow.runIndex,
                            ...bestRow
                        };
                    }
                    return { value: avgReturn, count: matchingRows.length };
                }
                return { value: null, count: 0 };
            })
        );

        return { cells, xValues, yValues, optimalConfig };
    }, [data, filters]);

    // Load returns data when optimal configuration changes
    useEffect(() => {
        if (optimalConfig) {
            console.log(`Loading returns for run index ${optimalConfig.runIndex}`);
            fetch(`Optimization Results/returns/returns_${optimalConfig.runIndex}.json`)
                .then(response => response.json())
                .then(data => {
                    console.log('Loaded returns data:', data);
                    setReturnsData(data);
                })
                .catch(err => console.error('Error loading returns:', err));
        }
    }, [optimalConfig]);

    if (loading) return React.createElement('div', { className: 'p-8 text-center' }, 'Loading...');
    if (error) return React.createElement('div', { className: 'p-8 text-center text-red-600' }, error);

    const formatProfitTarget = value => `${((value - 1) * 100).toFixed(1)}%`;
    const formatLossTarget = value => `${((1 - value) * 100).toFixed(1)}%`;

    // Main render remains the same...
    return React.createElement('div', { className: 'p-4 max-w-6xl mx-auto' },
        // ... rest of the render code remains the same ...

        // Title
        React.createElement('h1', { className: 'text-2xl font-bold mb-6' }, 'Optimization Results'),

        // Filters
        React.createElement('div', { className: 'grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4 mb-6' },
            React.createElement(Select, {
                label: 'MG Positive Signal',
                value: filters.use_mg_pos_signal,
                onChange: (value) => setFilters(prev => ({ ...prev, use_mg_pos_signal: value })),
                options: [
                    { value: '', label: 'Any' },
                    { value: 'true', label: 'True' },
                    { value: 'false', label: 'False' }
                ]
            }),
            React.createElement(Select, {
                label: 'MG 40 Signal',
                value: filters.use_mg_40_signal,
                onChange: (value) => setFilters(prev => ({ ...prev, use_mg_40_signal: value })),
                options: [
                    { value: '', label: 'Any' },
                    { value: 'true', label: 'True' },
                    { value: 'false', label: 'False' }
                ]
            }),
            React.createElement(Select, {
                label: 'SP500 Signal',
                value: filters.use_sp500_signal,
                onChange: (value) => setFilters(prev => ({ ...prev, use_sp500_signal: value })),
                options: [
                    { value: '', label: 'Any' },
                    { value: 'true', label: 'True' },
                    { value: 'false', label: 'False' }
                ]
            }),
            React.createElement(Select, {
                label: 'Sector Signals',
                value: filters.use_sector_signals,
                onChange: (value) => setFilters(prev => ({ ...prev, use_sector_signals: value })),
                options: [
                    { value: '', label: 'Any' },
                    { value: 'true', label: 'True' },
                    { value: 'false', label: 'False' }
                ]
            }),
            React.createElement(Select, {
                label: 'Entry Time Offset',
                value: filters.entry_time_offset,
                onChange: (value) => setFilters(prev => ({ ...prev, entry_time_offset: value })),
                options: [
                    { value: '', label: 'Any' },
                    ...Array.from(new Set(data.map(d => d.entry_time_offset))).sort().map(value => ({
                        value: value.toString(),
                        label: `${value} min`
                    }))
                ]
            })
        ),

        // Heatmap
        React.createElement('div', { className: 'bg-white rounded-lg shadow p-4 mb-6' },
            React.createElement('h2', { className: 'text-xl font-bold mb-4' }, 'Profit vs Loss Target Heatmap'),
            React.createElement('div', { className: 'overflow-x-auto' },
                React.createElement('div', { className: 'min-w-[600px]' },
                    // X-axis labels
                    React.createElement('div', { className: 'flex ml-20' },
                        xValues.map((value, i) => React.createElement('div', {
                            key: i,
                            className: 'flex-1 transform -rotate-45 origin-left pl-4 h-12 flex items-end'
                        }, formatProfitTarget(value)))
                    ),
                    // Heatmap cells
                    cells.map((row, y) => React.createElement('div', {
                        key: y,
                        className: 'flex'
                    },
                        React.createElement('div', {
                            className: 'w-20 flex items-center justify-end pr-2 text-sm font-medium'
                        }, formatLossTarget(yValues[y])),
                        React.createElement('div', {
                            className: 'flex-1 flex'
                        }, row.map((cell, x) => React.createElement(HeatmapCell, {
                            key: x,
                            value: cell.value,
                            count: cell.count,
                            min: Math.min(...cells.flat().filter(c => c.value !== null).map(c => c.value)),
                            max: Math.max(...cells.flat().filter(c => c.value !== null).map(c => c.value)),
                            isOptimal: optimalConfig &&
                                     xValues[x] === optimalConfig.profit_target &&
                                     yValues[y] === optimalConfig.loss_target
                        })))
                    ))
                )
            )
        ),

        // Best configuration details
        optimalConfig && React.createElement('div', { className: 'bg-white rounded-lg shadow p-4 mb-6' },
            React.createElement('h2', { className: 'text-xl font-bold mb-4' }, 'Best Configuration'),
            React.createElement('div', { className: 'grid grid-cols-2 gap-4' },
                React.createElement('div', null,
                    React.createElement('p', null, `Profit Target: ${formatProfitTarget(optimalConfig.profit_target)}`),
                    React.createElement('p', null, `Loss Target: ${formatLossTarget(optimalConfig.loss_target)}`),
                    React.createElement('p', null, `Return: ${optimalConfig.return.toFixed(1)}%`)
                )
            )
        ),

        // Returns plot
        returnsData && React.createElement('div', { className: 'bg-white rounded-lg shadow p-4' },
            React.createElement(ReturnsPlot, { data: returnsData })
        )
    );
};

// Render the application
ReactDOM.render(
    React.createElement(OptimizationHeatmaps),
    document.getElementById('root')
);