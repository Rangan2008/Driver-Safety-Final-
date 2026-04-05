import React from 'react';

const ModelShowcase = () => {
  return (
    <div style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%' }}>
      <iframe
        src="http://localhost:8501"
        width="100%"
        height="100%"
        style={{ border: 'none' }}
        title="Streamlit UI"
      />
    </div>
  );
};

export default ModelShowcase;
