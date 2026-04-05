import React from 'react';
import { motion } from 'motion/react';
import { Eye, AlertTriangle, HeartPulse, Activity } from 'lucide-react';

const Problem = () => {
  return (
    <section id="problem" className="py-24 relative bg-surface/30 border-y border-white/5">
      <div className="container mx-auto px-6">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <h2 className="text-3xl md:text-5xl font-bold mb-6">The Problem: Why Current Systems Fail</h2>
          <p className="text-white/60 text-lg">
            Current ecosystems fundamentally treat behavioral anomalies and physiological anomalies as mutually exclusive domains, creating a profound integration gap.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto">
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="p-8 rounded-2xl bg-background border border-white/10 relative overflow-hidden"
          >
            <div className="absolute top-0 right-0 p-6 opacity-10">
              <Eye className="w-32 h-32" />
            </div>
            <div className="w-12 h-12 rounded-xl bg-white/5 flex items-center justify-center mb-6 text-white/80">
              <AlertTriangle className="w-6 h-6" />
            </div>
            <h3 className="text-2xl font-semibold mb-4">Isolated Silos</h3>
            <p className="text-white/60 leading-relaxed">
              Existing vehicular safety systems traditionally operate in isolated silos, focusing on either visual behavior tracking or standalone physiological monitoring. They fail to see the complete picture of driver health and attention.
            </p>
          </motion.div>

          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="p-8 rounded-2xl bg-background border border-white/10 relative overflow-hidden"
          >
            <div className="absolute top-0 right-0 p-6 opacity-10">
              <HeartPulse className="w-32 h-32" />
            </div>
            <div className="w-12 h-12 rounded-xl bg-alert/10 flex items-center justify-center mb-6 text-alert">
              <Activity className="w-6 h-6" />
            </div>
            <h3 className="text-2xl font-semibold mb-4">Silent Medical Events</h3>
            <p className="text-white/60 leading-relaxed">
              If a driver experiences a severe, silent internal medical event—such as sudden extreme bradycardia—a purely visual system will fail to intervene until physical control of the vehicle is entirely lost.
            </p>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default Problem;
