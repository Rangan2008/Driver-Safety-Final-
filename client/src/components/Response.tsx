import React from 'react';
import { motion } from 'motion/react';
import { Network, Siren, Car } from 'lucide-react';

const Response = () => {
  return (
    <section id="response" className="py-24 bg-surface/30 border-y border-white/5">
      <div className="container mx-auto px-6">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <h2 className="text-3xl md:text-5xl font-bold mb-6">How It Works: Multi-Tiered Emergency Response</h2>
          <p className="text-white/60 text-lg">
            The architecture is explicitly designed around an active, three-tier intervention philosophy to ensure maximum safety.
          </p>
        </div>

        <div className="max-w-4xl mx-auto relative">
          {/* Connecting Line */}
          <div className="hidden md:block absolute left-[39px] top-10 bottom-10 w-0.5 bg-white/10" />

          <div className="space-y-12">
            {/* Step 1 */}
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="relative flex flex-col md:flex-row gap-8"
            >
              <div className="w-20 h-20 rounded-2xl bg-surface border border-white/10 flex items-center justify-center shrink-0 z-10 relative">
                <div className="absolute inset-0 bg-primary/10 rounded-2xl animate-pulse" />
                <Network className="w-8 h-8 text-primary relative z-10" />
              </div>
              <div className="pt-2">
                <div className="text-primary font-mono text-sm font-bold tracking-wider uppercase mb-2">Tier 1</div>
                <h3 className="text-2xl font-bold mb-3">Detect</h3>
                <p className="text-white/60 leading-relaxed text-lg">
                  The Central Decision Logic evaluates combined outputs from both the visual and health modules to identify anomalies in real-time.
                </p>
              </div>
            </motion.div>

            {/* Step 2 */}
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2 }}
              className="relative flex flex-col md:flex-row gap-8"
            >
              <div className="w-20 h-20 rounded-2xl bg-surface border border-white/10 flex items-center justify-center shrink-0 z-10 relative">
                <div className="absolute inset-0 bg-alert/10 rounded-2xl animate-pulse" />
                <Siren className="w-8 h-8 text-alert relative z-10" />
              </div>
              <div className="pt-2">
                <div className="text-alert font-mono text-sm font-bold tracking-wider uppercase mb-2">Tier 2</div>
                <h3 className="text-2xl font-bold mb-3">Alert (Immediate Intervention)</h3>
                <p className="text-white/60 leading-relaxed text-lg">
                  Upon detecting distraction or drowsiness, the system autonomously activates localized physical hardware, triggering in-cabin auditory sirens and flashing red warning lights to aggressively reawaken the driver.
                </p>
              </div>
            </motion.div>

            {/* Step 3 */}
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.4 }}
              className="relative flex flex-col md:flex-row gap-8"
            >
              <div className="w-20 h-20 rounded-2xl bg-surface border border-white/10 flex items-center justify-center shrink-0 z-10 relative">
                <div className="absolute inset-0 bg-success/10 rounded-2xl animate-pulse" />
                <Car className="w-8 h-8 text-success relative z-10" />
              </div>
              <div className="pt-2">
                <div className="text-success font-mono text-sm font-bold tracking-wider uppercase mb-2">Tier 3</div>
                <h3 className="text-2xl font-bold mb-3">Prevent and Rescue</h3>
                <p className="text-white/60 leading-relaxed text-lg">
                  In the event of a severe medical emergency or incapacitation, the system initiates an automated deceleration logic. Simultaneously, it compiles the driver's vital statistics and live GPS coordinates, dispatching an SOS notification to local authorities and emergency medical contacts.
                </p>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Response;
