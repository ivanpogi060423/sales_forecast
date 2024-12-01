import React, { useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Box, Button, CircularProgress, Typography } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { preprocessData, createDataset, denormalizeQuantity, generateFutureDates } from '../utils/dataPreprocessing';

const SalesPrediction = ({ data }) => {
    const [isTraining, setIsTraining] = useState(false);
    const [predictions, setPredictions] = useState(null);
    const [error, setError] = useState('');
    const [trainingProgress, setTrainingProgress] = useState({ epoch: 0, loss: 0, totalEpochs: 100 });

    const createModel = () => {
        const model = tf.sequential();
        
        model.add(tf.layers.lstm({
            units: 32,
            inputShape: [6, 2],
            returnSequences: false
        }));
        
        model.add(tf.layers.dense({
            units: 16,
            activation: 'relu'
        }));
        
        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid'
        }));

        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError'
        });

        return model;
    };

    const trainModel = async () => {
        setIsTraining(true);
        setError('');
        
        try {
            const processedData = preprocessData(data);
            const dataset = createDataset(
                processedData.dates,
                processedData.products,
                processedData.quantities
            );

            const model = createModel();

            await model.fit(dataset.inputs, dataset.outputs, {
                epochs: 100,
                batchSize: 32,
                validationSplit: 0.2,
                shuffle: true,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        setTrainingProgress(prev => ({
                            ...prev,
                            epoch: epoch + 1,
                            loss: logs.loss.toFixed(4)
                        }));
                        console.log(`Epoch ${epoch + 1}/100, Loss: ${logs.loss.toFixed(4)}`);
                    }
                }
            });

            // Generate predictions for next 6 months
            const lastDate = Math.max(...processedData.dates);
            const futureDates = Array.from({ length: 6 }, (_, i) => lastDate + i + 1);
            
            const predictions = [];
            const uniqueProducts = Object.keys(processedData.productEncoder);

            console.log('Generating predictions for products:', uniqueProducts);

            for (const product of uniqueProducts) {
                const productId = processedData.productEncoder[product];
                const inputSequence = futureDates.map(date => [date, productId]);
                
                // Create input tensor with correct shape [1, 6, 2]
                const inputTensor = tf.tensor3d([inputSequence], [1, 6, 2]);
                const prediction = model.predict(inputTensor);
                const predictionValues = Array.from(prediction.dataSync());
                
                console.log(`Predictions for ${product}:`, predictionValues);
                
                const denormalizedPredictions = predictionValues.map(value => 
                    denormalizeQuantity(value, processedData.minQuantity, processedData.maxQuantity)
                );

                console.log(`Denormalized predictions for ${product}:`, denormalizedPredictions);

                const futureDateStrings = generateFutureDates(
                    new Date(Math.max(...data.map(d => d.sales_date + '-01'))),
                    6
                );

                predictions.push({
                    product,
                    predictions: futureDateStrings.map((date, i) => ({
                        date,
                        quantity: denormalizedPredictions[i]
                    }))
                });

                // Clean up tensors
                inputTensor.dispose();
                prediction.dispose();
            }

            console.log('Final predictions:', predictions);
            setPredictions(predictions);
        } catch (err) {
            setError('Error training model: ' + err.message);
        } finally {
            setIsTraining(false);
        }
    };

    const renderChart = (productData) => {
        console.log('Rendering chart for product:', productData);
        
        const chartData = [
            ...data
                .filter(d => d.product_description === productData.product)
                .map(d => ({
                    date: d.sales_date,
                    actual: parseInt(d.quantity_sold),
                    predicted: null
                })),
            ...productData.predictions.map(p => ({
                date: p.date,
                actual: null,
                predicted: p.quantity
            }))
        ];

        console.log('Chart data:', chartData);

        return (
            <Box key={productData.product} sx={{ mt: 4, p: 2, border: '1px solid #e0e0e0', borderRadius: 2 }}>
                <Typography variant="h6" gutterBottom>
                    {productData.product} - Sales Forecast
                </Typography>
                <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="textSecondary">
                        Latest actual sale: {
                            chartData.filter(d => d.actual !== null)
                                .sort((a, b) => b.date.localeCompare(a.date))[0]?.actual || 'N/A'
                        } units
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                        Latest prediction: {
                            chartData.filter(d => d.predicted !== null)
                                .sort((a, b) => b.date.localeCompare(a.date))[0]?.predicted || 'N/A'
                        } units
                    </Typography>
                </Box>
                <LineChart width={800} height={400} data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                        dataKey="date"
                        tick={{ angle: -45 }}
                        height={70}
                        interval={0}
                    />
                    <YAxis />
                    <Tooltip content={({ active, payload, label }) => {
                        if (active && payload && payload.length) {
                            return (
                                <Box sx={{ bgcolor: 'white', p: 1, border: '1px solid #ccc' }}>
                                    <Typography variant="body2">{label}</Typography>
                                    {payload.map((entry) => (
                                        <Typography 
                                            key={entry.name}
                                            variant="body2"
                                            sx={{ color: entry.color }}
                                        >
                                            {entry.name}: {entry.value} units
                                        </Typography>
                                    ))}
                                </Box>
                            );
                        }
                        return null;
                    }}/>
                    <Legend />
                    <Line 
                        type="monotone" 
                        dataKey="actual" 
                        stroke="#8884d8" 
                        name="Actual Sales" 
                        strokeWidth={2}
                        dot={{ r: 4 }}
                    />
                    <Line 
                        type="monotone" 
                        dataKey="predicted" 
                        stroke="#82ca9d" 
                        name="Predicted Sales"
                        strokeWidth={2}
                        strokeDasharray="5 5"
                        dot={{ r: 4 }}
                    />
                </LineChart>
            </Box>
        );
    };

    return (
        <Box sx={{ my: 4 }}>
            <Button 
                variant="contained" 
                onClick={trainModel} 
                disabled={isTraining || !data.length}
            >
                {isTraining ? 'Training Model...' : 'Train Model'}
            </Button>

            {isTraining && (
                <Box sx={{ mt: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <CircularProgress size={24} sx={{ mr: 1 }} />
                        <Typography>Training model...</Typography>
                    </Box>
                    <Box sx={{ width: '100%', maxWidth: 400 }}>
                        <Typography variant="body2" color="textSecondary">
                            Epoch: {trainingProgress.epoch}/{trainingProgress.totalEpochs} 
                            {trainingProgress.loss > 0 && ` | Loss: ${trainingProgress.loss}`}
                        </Typography>
                        <Box sx={{ width: '100%', bgcolor: 'grey.200', height: 10, borderRadius: 1, mt: 1 }}>
                            <Box
                                sx={{
                                    width: `${(trainingProgress.epoch / trainingProgress.totalEpochs) * 100}%`,
                                    height: '100%',
                                    bgcolor: 'primary.main',
                                    borderRadius: 1,
                                    transition: 'width 0.3s'
                                }}
                            />
                        </Box>
                    </Box>
                </Box>
            )}

            {error && (
                <Typography color="error" sx={{ mt: 2 }}>
                    {error}
                </Typography>
            )}

            {predictions && predictions.length > 0 && (
                <Box sx={{ mt: 4 }}>
                    <Typography variant="h5" sx={{ mb: 2 }}>
                        Sales Predictions for Next 6 Months
                    </Typography>
                    {predictions.map(renderChart)}
                </Box>
            )}
        </Box>
    );
};

export default SalesPrediction;
