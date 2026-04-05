import React from 'react';
import { motion } from 'motion/react';
import { Eye, HeartPulse, Activity, Brain } from 'lucide-react';

const Solution = () => {
  const features = [
    {
      icon: <Eye className="w-6 h-6" />,
      title: "Real-Time Behavioral Tracking",
      description: "Deploys deep Convolutional Neural Networks (CNNs) for real-time visual extraction of spatial behavioral cues. Autonomously learns hierarchical spatial representations to classify cognitive states (normal driving, distraction, drowsiness).",
      color: "text-primary",
      bg: "bg-primary/10"
    },
    {
      icon: <HeartPulse className="w-6 h-6" />,
      title: "Continuous Physiological Surveillance",
      description: "Leverages robust Machine Learning algorithms (Support Vector Machines with RBF kernels) for continuous physiological risk prediction. Maps real-time heart rates against strict biological thresholds.",
      color: "text-alert",
      bg: "bg-alert/10"
    },
    {
      icon: <Activity className="w-6 h-6" />,
      title: "Background Cardiovascular Risk Assessment",
      description: "A Random Forest classifier evaluates historical or static driver health data, such as age and resting blood pressure, to construct a comprehensive baseline risk profile.",
      color: "text-blue-400",
      bg: "bg-blue-400/10"
    },
    {
      icon: <Brain className="w-6 h-6" />,
      title: "Exceptional Predictive Precision",
      description: "Tested across specialized datasets, the visual behavioral tracking achieved a 100.00% accuracy rate, successfully isolating the visual geometry of distraction from standard driving postures.",
      color: "text-success",
      bg: "bg-success/10"
    }
  ];

  return (
    <section id="solution" className="py-24 relative">
      <div className="container mx-auto px-6">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <h2 className="text-3xl md:text-5xl font-bold mb-6">Our Solution: Core Technologies & Features</h2>
          <p className="text-white/60 text-lg">Bridging the integration gap with state-of-the-art machine learning and neural networks.</p>
        </div>

        <div className="grid md:grid-cols-2 gap-6 max-w-5xl mx-auto">
          {features.map((feature, idx) => (
            <motion.div 
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: idx * 0.1 }}
              className="p-8 rounded-2xl bg-surface border border-white/5 hover:border-white/20 transition-colors group"
            >
              <div className={`w-14 h-14 rounded-xl ${feature.bg} ${feature.color} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform`}>
                {feature.icon}
              </div>
              <h3 className="text-2xl font-semibold mb-4">{feature.title}</h3>
              <p className="text-white/60 leading-relaxed">{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Solution;
