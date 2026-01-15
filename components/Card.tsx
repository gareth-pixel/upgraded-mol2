import React from 'react';

export const Card: React.FC<{ children: React.ReactNode; title?: string; className?: string }> = ({ children, title, className = '' }) => {
  return (
    <div className={`bg-white overflow-hidden shadow rounded-lg border border-gray-100 ${className}`}>
      {title && (
        <div className="px-5 py-4 border-b border-gray-100 bg-gray-50">
          <h3 className="text-lg leading-6 font-medium text-gray-900">{title}</h3>
        </div>
      )}
      <div className="px-5 py-5">
        {children}
      </div>
    </div>
  );
};