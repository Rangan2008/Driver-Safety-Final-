import React from 'react';

const CTA = () => {
  return (
    <section className="py-24 relative overflow-hidden">
      <div className="absolute inset-0 bg-primary/5" />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-3xl h-[400px] bg-primary/20 rounded-full blur-[100px] pointer-events-none" />
      
      <div className="container mx-auto px-6 relative z-10 text-center">
        <h2 className="text-4xl md:text-6xl font-bold mb-6">Experience the Future of Safety</h2>
        <p className="text-xl text-white/70 mb-10 max-w-2xl mx-auto">
          Deploy our integrated computational framework to protect your drivers with 100% predictive precision.
        </p>
        <button className="px-10 py-5 bg-primary text-background font-bold text-lg rounded-full hover:bg-primary/90 transition-transform hover:scale-105">
          Request a Technical Demo
        </button>
      </div>
    </section>
  );
};

export default CTA;
