import React, { useState, useRef } from 'react';
import { createPortal } from 'react-dom';

/**
 * Unified portal-based tooltip with optional interactive mode.
 *
 * @param {Object} props
 * @param {React.ReactNode} props.content - Tooltip body
 * @param {React.ReactNode} props.children - Trigger element
 * @param {string} [props.className] - CSS class for trigger wrapper
 * @param {boolean} [props.interactive] - When true, tooltip stays open while hovered (links, copy, etc.)
 */
export const PortalTooltip = ({ content, children, className = "inline-block", interactive = true }) => {
    const [visible, setVisible] = useState(false);
    const [coords, setCoords] = useState({ top: 0, left: 0 });
    const triggerRef = useRef(null);
    const tooltipRef = useRef(null);
    const isOverTrigger = useRef(false);
    const isOverTooltip = useRef(false);

    const checkHide = () => {
        if (!isOverTrigger.current && !isOverTooltip.current) {
            setVisible(false);
        }
    };

    const show = () => {
        isOverTrigger.current = true;
        if (triggerRef.current) {
            const rect = triggerRef.current.getBoundingClientRect();
            setCoords({ top: rect.top - 8, left: rect.left + (rect.width / 2) });
            setVisible(true);
        }
    };

    const hideTrigger = () => {
        isOverTrigger.current = false;
        if (interactive) {
            setTimeout(checkHide, 100);
        } else {
            setVisible(false);
        }
    };

    return (
        <React.Fragment>
            <div
                ref={triggerRef}
                onMouseEnter={show}
                onMouseLeave={hideTrigger}
                className={className}
                style={className?.includes('contents') ? { display: 'contents' } : undefined}
            >
                {children}
            </div>
            {visible && createPortal(
                <div
                    ref={tooltipRef}
                    onMouseEnter={interactive ? () => { isOverTooltip.current = true; } : undefined}
                    onMouseLeave={interactive ? () => { isOverTooltip.current = false; setTimeout(checkHide, 100); } : undefined}
                    className={`fixed z-[99999] animate-in fade-in zoom-in-95 duration-100 ${interactive ? 'pointer-events-auto' : 'pointer-events-none'}`}
                    style={{
                        top: coords.top,
                        left: coords.left,
                        transform: 'translate(-50%, -100%)',
                        filter: interactive ? 'drop-shadow(0 4px 20px rgba(0,0,0,0.5))' : undefined
                    }}
                >
                    {content}
                </div>,
                document.body
            )}
        </React.Fragment>
    );
};
