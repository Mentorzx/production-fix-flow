import { createContext, useContext, useEffect, useState } from 'react';

const ThemeContext = createContext();

export const useTheme = () => {
    const context = useContext(ThemeContext);
    if (!context) {
        throw new Error('useTheme must be used within a ThemeProvider');
    }
    return context;
};

export const ThemeProvider = ({ children }) => {
    const [theme, setTheme] = useState(() => {
        // Check localStorage first
        const savedTheme = localStorage.getItem('pff-theme');
        if (savedTheme) return savedTheme;

        // Then check system preference
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
            return 'light';
        }
        return 'dark'; // Default SOTA Dark Mode
    });

    useEffect(() => {
        const root = window.document.documentElement;

        // Remove old theme class if any (though we use data-theme now)
        root.classList.remove('light', 'dark');
        root.classList.add(theme);

        // Set data-attribute for CSS selectors
        root.setAttribute('data-theme', theme);

        // Persist to localStorage
        localStorage.setItem('pff-theme', theme);
    }, [theme]);

    const toggleTheme = () => {
        setTheme(prev => prev === 'dark' ? 'light' : 'dark');
    };

    return (
        <ThemeContext.Provider value={{ theme, toggleTheme }}>
            {children}
        </ThemeContext.Provider>
    );
};
