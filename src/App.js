import React, { useState } from 'react';
import { Container, Typography, Box } from '@mui/material';
import FileUpload from './components/FileUpload';
import SalesPrediction from './components/SalesPrediction';

function App() {
  const [data, setData] = useState([]);

  const handleDataLoaded = (salesData) => {
    setData(salesData);
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Sales Prediction Dashboard
        </Typography>
        <FileUpload onDataLoaded={handleDataLoaded} />
        {data.length > 0 && <SalesPrediction data={data} />}
      </Box>
    </Container>
  );
}

export default App;
