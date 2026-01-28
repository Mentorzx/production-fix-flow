import { useState, useMemo, useCallback, useRef } from 'react';
import { SortedTableHeader, PaginationControls } from "./TableComponents.jsx";

export const SortableTable = ({ data, columns, defaultSort = { key: 'id', direction: 'desc' }, className = "" }) => {
    const [sort, setSort] = useState(defaultSort);
    const [currentPage, setCurrentPage] = useState(1);
    const [rowsPerPage, setRowsPerPage] = useState(10);
    const containerRef = useRef(null);

    const sortedData = useMemo(() => {
        if (!data) return [];
        let sorted = [...data];
        if (sort && sort.key) {
            const activeColumn = columns.find(c => c.key === sort.key);
            sorted.sort((a, b) => {
                let aVal, bVal;
                if (activeColumn?.sortValue) { aVal = activeColumn.sortValue(a); bVal = activeColumn.sortValue(b); }
                else { aVal = a[sort.key]; bVal = b[sort.key]; }
                if (aVal === bVal) return 0;
                if (aVal === null || aVal === undefined) return 1;
                if (bVal === null || bVal === undefined) return -1;
                return sort.direction === 'asc' ? (aVal < bVal ? -1 : 1) : (aVal < bVal ? 1 : -1);
            });
        }
        return sorted;
    }, [data, sort, columns]);

    const paginatedData = useMemo(() => {
        if (rowsPerPage === 'All') return sortedData;
        const start = (currentPage - 1) * rowsPerPage;
        return sortedData.slice(start, start + rowsPerPage);
    }, [sortedData, currentPage, rowsPerPage]);

    const onSort = useCallback((newSort) => {
        const newKey = typeof newSort === 'string' ? newSort : newSort?.key;
        setSort(prev => {
            if (prev && prev.key === newKey) {
                return prev.direction === 'desc' ? { key: newKey, direction: 'asc' } : defaultSort;
            }
            return { key: newKey, direction: 'desc' };
        });
    }, [defaultSort]);

    return (
        <div className={`flex flex-col h-full rounded-xl shadow-xl overflow-hidden ${className}`} style={{ backgroundColor: 'var(--viz-bg-canvas)', borderColor: 'var(--viz-border)', borderWidth: '1px' }}>
            <div ref={containerRef} className="flex-1 overflow-auto custom-scrollbar relative">
                <table className="w-full text-left border-collapse table-fixed tabular-nums">
                    <thead className="sticky top-0 z-20 shadow-sm" style={{ backgroundColor: 'var(--viz-bg-surface)' }}>
                        <tr>
                            {columns.map((col, idx) => {
                                const prevGroup = idx > 0 ? columns[idx - 1]?.group : null;
                                const isGroupStart = idx > 0 && col.group && col.group !== prevGroup;
                                const paddingClass = idx === 0
                                    ? 'px-4'
                                    : isGroupStart
                                        ? 'pl-6 pr-3'
                                        : 'px-3';
                                const borderClass = isGroupStart ? 'border-l border-zinc-800/60' : '';

                                return (
                                    <th
                                        key={col.key || idx}
                                        className={`py-3 ${paddingClass} ${borderClass} ${idx === 0 ? 'sticky left-0 z-30 shadow-[2px_0_5px_-2px_rgba(0,0,0,0.1)]' : ''}`}
                                        style={{
                                            width: col.width || 'auto',
                                            backgroundColor: 'var(--viz-bg-surface)',
                                            borderBottom: '1px solid var(--viz-border)'
                                        }}
                                    >
                                        <SortedTableHeader
                                            label={col.label}
                                            sortKey={col.key}
                                            currentSort={sort}
                                            onSort={col.sortable ? onSort : undefined}
                                            helpText={col.helpText}
                                            directionHint={col.direction}
                                            align={col.align}
                                            isFirst={idx === 0}
                                            isLast={idx === columns.length - 1}
                                            className={idx === 0 ? 'border-r border-zinc-800/50' : ''}
                                        />
                                    </th>
                                );
                            })}
                        </tr>
                    </thead>
                    <tbody className="text-xs font-mono">
                        {paginatedData.map((row) => (
                            <tr
                                key={row.id}
                                className="group transition-colors"
                                style={{ borderBottom: '1px solid var(--viz-border)' }}
                            >
                                {columns.map((col, j) => {
                                    const isNumeric = typeof row[col.key] === 'number';
                                    const cellAlign = col.align || (isNumeric ? 'right' : 'left');
                                    const prevGroup = j > 0 ? columns[j - 1]?.group : null;
                                    const isGroupStart = j > 0 && col.group && col.group !== prevGroup;
                                    const paddingClass = j === 0
                                        ? 'px-4'
                                        : isGroupStart
                                            ? 'pl-6 pr-3'
                                            : 'px-3';
                                    const borderClass = isGroupStart ? 'border-l border-zinc-800/60' : '';

                                    return (
                                        <td
                                            key={col.key || j}
                                            className={`py-2 ${paddingClass} ${borderClass} truncate transition-colors group-hover:bg-white/5 ${j === 0 ? 'sticky left-0 z-10 shadow-[2px_0_5px_-2px_rgba(0,0,0,0.1)]' : ''}`}
                                            style={{
                                                backgroundColor: j === 0 ? 'var(--viz-bg-surface)' : 'transparent',
                                                whiteSpace: 'nowrap',
                                                textAlign: cellAlign,
                                                color: 'var(--viz-text-secondary)'
                                            }}
                                        >
                                            <div className="flex items-center h-full w-full" style={{ justifyContent: cellAlign === 'right' ? 'flex-end' : (cellAlign === 'center' ? 'center' : 'flex-start') }}>
                                                {col.render ? col.render(row[col.key], row) : (row[col.key] ?? <span className="text-zinc-600">—</span>)}
                                            </div>
                                        </td>
                                    );
                                })}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
            <div style={{ backgroundColor: 'var(--viz-bg-surface)', borderTop: '1px solid var(--viz-border)' }}>
                <PaginationControls
                    totalItems={sortedData.length}
                    currentPage={currentPage}
                    rowsPerPage={rowsPerPage}
                    onPageChange={setCurrentPage}
                    onRowsPerPageChange={(value) => {
                        setRowsPerPage(value);
                        setCurrentPage(1);
                    }}
                />
            </div>
        </div>
    );
};
