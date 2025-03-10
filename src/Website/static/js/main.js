/**
 * AutoTracker - Main JavaScript file
 * Contains utility functions for the website
 */

// Add responsive behavior for components
document.addEventListener('DOMContentLoaded', () => {
    // Add card hover effects to all elements with card-hover class
    const cards = document.querySelectorAll('.card-hover');
    cards.forEach(card => {
        card.classList.add('transition-all');
    });

    // Add animation to status indicators
    const statusIndicators = document.querySelectorAll('.status-indicator');
    statusIndicators.forEach(indicator => {
        if (indicator.classList.contains('active')) {
            indicator.classList.add('status-active');
        }
    });

    // Responsive navigation menu toggle for mobile
    const navToggle = document.getElementById('nav-toggle');
    const navMenu = document.getElementById('nav-menu');
    
    if (navToggle && navMenu) {
        navToggle.addEventListener('click', () => {
            navMenu.classList.toggle('hidden');
        });
    }

    // Theme toggle functionality (if added in the future)
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            document.documentElement.classList.toggle('dark');
            localStorage.setItem('theme', document.documentElement.classList.contains('dark') ? 'dark' : 'light');
        });
    }
});

// Utility function to format date/time
function formatDateTime(date) {
    return new Intl.DateTimeFormat('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    }).format(date);
}

// Utility function to format numbers with proper units
function formatValue(value, unit, precision = 2) {
    return `${value.toFixed(precision)} ${unit}`;
}

// Utility function for responsive chart resizing
function resizeChart(chart) {
    if (chart) {
        chart.resize();
    }
}

// Window resize event for responsiveness
window.addEventListener('resize', () => {
    // Your responsive behavior here
    // This can be used to resize charts or adjust UI elements
});

// Utility function for API requests
async function fetchAPI(endpoint, options = {}) {
    try {
        const response = await fetch(endpoint, options);
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('API request error:', error);
        return null;
    }
}

// Utility function to download data as CSV
function downloadCSV(data, filename = 'autotracker-data.csv') {
    let csvContent = "data:text/csv;charset=utf-8,";
    
    // Add headers
    const headers = Object.keys(data[0]).join(',');
    csvContent += headers + "\r\n";
    
    // Add data rows
    data.forEach(row => {
        const values = Object.values(row).join(',');
        csvContent += values + "\r\n";
    });
    
    // Create download link
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}