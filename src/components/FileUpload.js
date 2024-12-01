import React, { useState } from 'react';
import Papa from 'papaparse';
import { Button, Box, Alert } from '@mui/material';

const FileUpload = ({ onDataLoaded }) => {
    const [error, setError] = useState('');

    const handleFileUpload = (event) => {
        const file = event.target.files[0];
        if (!file) return;

        if (file.type !== 'text/csv') {
            setError('Please upload a CSV file');
            return;
        }

        Papa.parse(file, {
            header: true,
            skipEmptyLines: true,
            dynamicTyping: true,
            complete: (results) => {
                if (results.errors.length > 0) {
                    console.error('CSV parsing errors:', results.errors);
                    setError(`Error parsing CSV file: ${results.errors[0].message}`);
                    return;
                }

                // Filter out any empty rows
                const validRows = results.data.filter(row => 
                    row.sales_date && 
                    row.product_description && 
                    row.quantity_sold !== null && 
                    row.quantity_sold !== undefined
                );

                if (validRows.length === 0) {
                    setError('No valid data found in the CSV file');
                    return;
                }

                const requiredColumns = ['sales_date', 'product_description', 'quantity_sold'];
                const headers = Object.keys(validRows[0]);
                const missingColumns = requiredColumns.filter(col => !headers.includes(col));

                if (missingColumns.length > 0) {
                    setError(`Missing required columns: ${missingColumns.join(', ')}`);
                    return;
                }

                // Validate data format
                const validData = validRows.filter(row => 
                    /^\d{4}-\d{2}$/.test(row.sales_date) && 
                    typeof row.product_description === 'string' &&
                    !isNaN(parseFloat(row.quantity_sold))
                );

                if (validData.length === 0) {
                    setError('No valid data found in the CSV file');
                    return;
                }

                setError('');
                onDataLoaded(validData);
            },
            error: (error) => {
                console.error('File reading error:', error);
                setError('Error reading file: ' + error.message);
            }
        });
    };

    return (
        <Box sx={{ my: 2 }}>
            <input
                accept=".csv"
                style={{ display: 'none' }}
                id="raised-button-file"
                type="file"
                onChange={handleFileUpload}
            />
            <label htmlFor="raised-button-file">
                <Button variant="contained" component="span">
                    Upload CSV File
                </Button>
            </label>
            {error && (
                <Alert severity="error" sx={{ mt: 2 }}>
                    {error}
                </Alert>
            )}
        </Box>
    );
};

export default FileUpload;
