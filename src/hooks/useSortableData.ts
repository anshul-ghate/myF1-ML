import { useState, useMemo } from 'react';

type SortConfig<T> = {
  key: keyof T;
  direction: 'ascending' | 'descending';
} | null;

export const useSortableData = <T>(
  items: T[],
  initialSortConfig: SortConfig<T> = null
) => {
  const [sortConfig, setSortConfig] = useState<SortConfig<T>>(initialSortConfig);

  const sortedItems = useMemo(() => {
    if (!items) {
      return [];
    }
    const sortableItems = [...items];
    if (sortConfig !== null) {
      sortableItems.sort((a, b) => {
        const aValue = a[sortConfig.key];
        const bValue = b[sortConfig.key];

        if (aValue < bValue) {
          return sortConfig.direction === 'ascending' ? -1 : 1;
        }
        if (aValue > bValue) {
          return sortConfig.direction === 'ascending' ? 1 : -1;
        }
        return 0;
      });
    }
    return sortableItems;
  }, [items, sortConfig]);

  const requestSort = (key: keyof T) => {
    let direction: 'ascending' | 'descending' = 'ascending';
    if (
      sortConfig &&
      sortConfig.key === key &&
      sortConfig.direction === 'ascending'
    ) {
      direction = 'descending';
    }
    setSortConfig({ key, direction });
  };

  return { items: sortedItems, requestSort, sortConfig };
};