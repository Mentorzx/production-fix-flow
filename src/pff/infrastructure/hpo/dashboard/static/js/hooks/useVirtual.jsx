import { useState, useEffect, useMemo, useCallback } from 'react';

export const useVirtual = ({ itemCount, itemHeight, overscan = 5, containerRef }) => {
    const [scrollTop, setScrollTop] = useState(0);
    const [containerHeight, setContainerHeight] = useState(0);

    const handleScroll = useCallback((e) => setScrollTop(e.target.scrollTop), []);

    useEffect(() => {
        const container = containerRef.current;
        if (!container) return;
        const resizeObserver = new ResizeObserver((entries) => {
            for (let entry of entries) setContainerHeight(entry.contentRect.height);
        });
        resizeObserver.observe(container);
        container.addEventListener('scroll', handleScroll);
        setContainerHeight(container.offsetHeight);
        return () => {
            resizeObserver.disconnect();
            container.removeEventListener('scroll', handleScroll);
        };
    }, [containerRef, handleScroll]);

    return useMemo(() => {
        const start = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan);
        const end = Math.min(itemCount - 1, Math.floor((scrollTop + containerHeight) / itemHeight) + overscan);
        const virtualItems = [];
        for (let i = start; i <= end; i++) { virtualItems.push({ index: i, offsetTop: i * itemHeight }); }
        return { virtualItems, totalHeight: itemCount * itemHeight };
    }, [itemCount, itemHeight, overscan, scrollTop, containerHeight]);
};
