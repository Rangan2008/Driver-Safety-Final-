import React from 'react';
import { motion } from 'motion/react';
import { ChevronRight, HeartPulse, Activity, Eye, Network } from 'lucide-react';

const Hero = ({ setCurrentView }: { setCurrentView: (v: 'home' | 'profile' | 'showcase') => void }) => {
  return (
    <section className="relative min-h-screen flex items-center justify-center pt-20 overflow-hidden">
      {/* Background glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-primary/20 rounded-full blur-[120px] opacity-50 pointer-events-none" />
      
      <div className="container mx-auto px-6 grid lg:grid-cols-2 gap-12 items-center relative z-10">
        <motion.div 
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="max-w-2xl"
        >
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 mb-6">
            <span className="w-2 h-2 rounded-full bg-success animate-pulse" />
            <span className="text-xs font-medium tracking-wide uppercase text-white/80">Active Integration</span>
          </div>
          <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold leading-[1.1] mb-6 tracking-tight">
            AI-Powered <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-blue-500">
              Driver Health & Safety
            </span>
            <br />Monitoring System.
          </h1>
          <p className="text-xl md:text-2xl text-white font-medium mb-4 leading-relaxed">
            Synthesizing biological health metrics with continuous computer vision to create a highly resilient, proactive safety net.
          </p>
          <p className="text-base text-white/60 mb-8 leading-relaxed border-l-2 border-primary/50 pl-4">
            Road traffic collisions remain a paramount global cause of fatalities, frequently precipitated by driver fatigue, cognitive distraction, and sudden physiological health crises. Our integrated computational framework breaks down isolated silos to simultaneously track vital health parameters and dynamic driving behavior.
          </p>
          <div className="flex flex-wrap items-center gap-4">
            <button 
              onClick={() => setCurrentView('showcase')}
              className="px-8 py-4 bg-primary text-background font-semibold rounded-full hover:bg-primary/90 transition-colors flex items-center gap-2"
            >
              Model Showcase <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </motion.div>

        {/* Interactive Graphic */}
        <div className="relative h-[500px] w-full flex items-center justify-center">
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 1, delay: 0.2 }}
            className="relative w-full max-w-md aspect-square rounded-full border border-white/10 bg-surface/50 backdrop-blur-sm flex items-center justify-center"
          >
            {/* Scanning line */}
            <motion.div 
              animate={{ top: ['0%', '100%', '0%'] }}
              transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
              className="absolute left-0 right-0 h-1 bg-primary/50 shadow-[0_0_20px_rgba(0,229,255,0.5)] z-20"
            />
            
            {/* Abstract Face/Head */}
            <div className="relative w-48 h-64 border-2 border-white/20 rounded-[100px] flex flex-col items-center justify-center gap-8 overflow-hidden">
               {/* Eyes */}
               <div className="flex gap-8">
                 <motion.div 
                   animate={{ height: ['8px', '2px', '8px'] }}
                   transition={{ duration: 4, repeat: Infinity, times: [0, 0.1, 0.2] }}
                   className="w-12 h-2 bg-primary rounded-full shadow-[0_0_15px_rgba(0,229,255,0.8)]" 
                 />
                 <motion.div 
                   animate={{ height: ['8px', '2px', '8px'] }}
                   transition={{ duration: 4, repeat: Infinity, times: [0, 0.1, 0.2] }}
                   className="w-12 h-2 bg-primary rounded-full shadow-[0_0_15px_rgba(0,229,255,0.8)]" 
                 />
               </div>
               {/* Heart/Vitals */}
               <HeartPulse className="w-10 h-10 text-alert animate-pulse" />
            </div>

            {/* Floating Metrics */}
            <motion.div 
              animate={{ y: [0, -10, 0] }}
              transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
              className="absolute top-10 -left-10 bg-surface border border-white/10 p-3 rounded-xl shadow-2xl flex items-center gap-3 backdrop-blur-md"
            >
              <div className="p-2 bg-alert/20 rounded-lg text-alert">
                <Activity className="w-5 h-5" />
              </div>
              <div>
                <div className="text-xs text-white/50">Physiological</div>
                <div className="font-mono font-bold">SVM Active</div>
              </div>
            </motion.div>

            <motion.div 
              animate={{ y: [0, 10, 0] }}
              transition={{ duration: 4, repeat: Infinity, ease: "easeInOut", delay: 1 }}
              className="absolute bottom-20 -right-10 bg-surface border border-white/10 p-3 rounded-xl shadow-2xl flex items-center gap-3 backdrop-blur-md"
            >
              <div className="p-2 bg-primary/20 rounded-lg text-primary">
                <Eye className="w-5 h-5" />
              </div>
              <div>
                <div className="text-xs text-white/50">Behavioral CNN</div>
                <div className="font-mono font-bold">Tracking</div>
              </div>
            </motion.div>
            
            <motion.div 
              animate={{ y: [0, -8, 0] }}
              transition={{ duration: 3.5, repeat: Infinity, ease: "easeInOut", delay: 0.5 }}
              className="absolute top-1/2 -right-12 bg-surface border border-white/10 p-3 rounded-xl shadow-2xl flex items-center gap-3 backdrop-blur-md"
            >
              <div className="p-2 bg-white/10 rounded-lg text-white">
                <Network className="w-5 h-5" />
              </div>
              <div>
                <div className="text-xs text-white/50">Integration</div>
                <div className="font-mono font-bold text-success">Synced</div>
              </div>
            </motion.div>

          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
