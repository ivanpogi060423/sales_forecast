import * as tf from '@tensorflow/tfjs';

export const preprocessData = (data) => {
    // Convert dates to numerical format (months since start)
    const dates = data.map(row => row.sales_date);
    const startDate = new Date(Math.min(...dates.map(d => new Date(d + '-01'))));
    
    const processedDates = dates.map(date => {
        const currentDate = new Date(date + '-01');
        const monthsDiff = (currentDate.getFullYear() - startDate.getFullYear()) * 12 
            + (currentDate.getMonth() - startDate.getMonth());
        return monthsDiff;
    });

    // Encode product descriptions
    const uniqueProducts = [...new Set(data.map(row => row.product_description))];
    const productEncoder = {};
    uniqueProducts.forEach((product, index) => {
        productEncoder[product] = index;
    });

    const encodedProducts = data.map(row => productEncoder[row.product_description]);

    // Normalize quantities
    const quantities = data.map(row => parseFloat(row.quantity_sold));
    const minQuantity = Math.min(...quantities);
    const maxQuantity = Math.max(...quantities);
    const normalizedQuantities = quantities.map(q => 
        (q - minQuantity) / (maxQuantity - minQuantity)
    );

    return {
        dates: processedDates,
        products: encodedProducts,
        quantities: normalizedQuantities,
        productEncoder,
        minQuantity,
        maxQuantity,
        startDate
    };
};

export const createDataset = (dates, products, quantities, windowSize = 6) => {
    const X = [];
    const y = [];

    for (let i = 0; i < dates.length - windowSize; i++) {
        const window = [];
        for (let j = 0; j < windowSize; j++) {
            window.push([dates[i + j], products[i + j]]);
        }
        X.push(window);
        y.push(quantities[i + windowSize]);
    }

    return {
        inputs: tf.tensor3d(X),
        outputs: tf.tensor2d(y, [y.length, 1])
    };
};

export const denormalizeQuantity = (normalizedValue, minQuantity, maxQuantity) => {
    return Math.round(normalizedValue * (maxQuantity - minQuantity) + minQuantity);
};

export const generateFutureDates = (startDate, numMonths) => {
    const dates = [];
    const currentDate = new Date(startDate);
    
    for (let i = 1; i <= numMonths; i++) {
        currentDate.setMonth(currentDate.getMonth() + 1);
        const year = currentDate.getFullYear();
        const month = String(currentDate.getMonth() + 1).padStart(2, '0');
        dates.push(`${year}-${month}`);
    }
    
    return dates;
};
