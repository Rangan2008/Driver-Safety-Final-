import React from 'react';
import { motion } from 'motion/react';
import { FileText, CheckCircle2 } from 'lucide-react';

const Docs = () => {
  return (
    <div className="pt-28 pb-24 container mx-auto px-6 max-w-4xl">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-12">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 mb-6">
          <FileText className="w-4 h-4 text-primary" />
          <span className="text-xs font-medium tracking-wide uppercase text-primary">Technical Documentation</span>
        </div>
        <h1 className="text-4xl md:text-5xl font-bold mb-6 tracking-tight">
          AI-Powered Drive Health and Safety Monitoring System
        </h1>
        <p className="text-xl text-white/60 leading-relaxed">
          A comprehensive overview of the system architecture, methodology, and performance metrics based on the official research thesis.
        </p>
      </motion.div>

      <div className="space-y-16">
        {/* Abstract Section */}
        <section>
          <h2 className="text-2xl font-bold mb-6 text-white border-b border-white/10 pb-4">1. Abstract & Problem Statement</h2>
          <div className="prose prose-invert max-w-none text-white/70 space-y-4">
            <p>
              Road traffic collisions remain a paramount global cause of fatalities, frequently precipitated by driver fatigue, cognitive distraction, and sudden physiological health crises. Existing vehicular safety systems traditionally operate in isolated silos, focusing on either visual behavior tracking or standalone physiological monitoring.
            </p>
            <p>
              This project addresses this critical vulnerability by proposing an integrated AI-Powered Driver Health and Safety Monitoring System designed to simultaneously track vital health parameters and dynamic driving behavior.
            </p>
          </div>
        </section>

        {/* Architecture Section */}
        <section>
          <h2 className="text-2xl font-bold mb-6 text-white border-b border-white/10 pb-4">2. System Architecture & Pipeline</h2>
          <div className="prose prose-invert max-w-none text-white/70 space-y-4">
            <p>
              The system is engineered as a highly modular, multi-threaded architecture. Rather than relying on a monolithic algorithmic structure, the system bifurcates the processing workload into two distinct, parallel pipelines:
            </p>
            <ul className="list-disc pl-6 space-y-2 text-white/80">
              <li><strong>Visual Driver Behavior Detection Module:</strong> Utilizes deep Convolutional Neural Networks (CNNs).</li>
              <li><strong>Physiological Health Monitoring Module:</strong> Utilizes Support Vector Machines (SVMs) and Random Forests.</li>
            </ul>
            <p>
              These parallel streams continuously feed real-time inferential data into a Central Decision and Alert Subsystem. The architecture is explicitly designed around a three-tier intervention philosophy: <strong>Detect, Alert, Prevent and Rescue</strong>.
            </p>
          </div>
        </section>

        {/* CNN Section */}
        <section>
          <h2 className="text-2xl font-bold mb-6 text-white border-b border-white/10 pb-4">3. Computer Vision Framework (CNN)</h2>
          <div className="prose prose-invert max-w-none text-white/70 space-y-4">
            <p>
              To accurately capture and classify complex human behaviors such as cognitive distraction, mobile phone usage, and drowsiness, the system deploys a deep CNN that autonomously learns hierarchical spatial representations from input video frames.
            </p>
            <div className="grid md:grid-cols-3 gap-4 mt-6">
              <div className="bg-surface border border-white/10 p-4 rounded-xl">
                <h4 className="font-semibold text-white mb-2">Normal Driving</h4>
                <p className="text-sm">Forward-facing, hands on the wheel, attentive gaze.</p>
              </div>
              <div className="bg-surface border border-white/10 p-4 rounded-xl">
                <h4 className="font-semibold text-white mb-2">Distracted Driving</h4>
                <p className="text-sm">Manual and cognitive distraction, mobile phone usage, looking at passengers.</p>
              </div>
              <div className="bg-surface border border-white/10 p-4 rounded-xl">
                <h4 className="font-semibold text-white mb-2">Drowsy State</h4>
                <p className="text-sm">Prolonged Eye Aspect Ratio (EAR) closures, frequent yawning or head-nodding.</p>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-xl p-6 mt-6">
              <h4 className="font-semibold text-primary mb-2 flex items-center gap-2"><CheckCircle2 className="w-5 h-5" /> Performance Validation</h4>
              <p className="text-sm text-white/80">
                When evaluated against the withheld validation imagery, the CNN model achieved a <strong>100.00% accuracy rate</strong>. The normalized confusion matrix confirmed absolute precision in distinguishing between normal driving and distracted states without generating any false positives or false negatives.
              </p>
            </div>
          </div>
        </section>

        {/* SVM Section */}
        <section>
          <h2 className="text-2xl font-bold mb-6 text-white border-b border-white/10 pb-4">4. Physiological Monitoring (SVM & RF)</h2>
          <div className="prose prose-invert max-w-none text-white/70 space-y-4">
            <p>
              While the CNN handles external behavior, the internal physiological state is monitored using a highly tuned Support Vector Machine (SVM). The core mathematical challenge is the non-linear nature of heart rate classification.
            </p>
            <ul className="list-disc pl-6 space-y-2 text-white/80">
              <li><strong>The RBF Kernel Trick:</strong> To overcome the limitations of linear models, the system utilizes an SVM equipped with a Radial Basis Function (RBF) kernel. This projects the 1D BPM data into a higher-dimensional space where a clear, non-linear separation can be achieved.</li>
              <li><strong>Hard Margin Enforcement:</strong> Configured with a significantly high penalty parameter (C=1000) to enforce a strict "Hard Margin," aggressively penalizing any misclassification and preventing dangerous boundary blurring.</li>
            </ul>

            <h3 className="text-xl font-semibold text-white mt-8 mb-4">Edge Case Boundary Testing</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse bg-surface rounded-xl overflow-hidden">
                <thead>
                  <tr className="bg-white/5">
                    <th className="p-4 text-white font-medium">Input BPM</th>
                    <th className="p-4 text-white font-medium">Physiological State</th>
                    <th className="p-4 text-white font-medium">Model Prediction</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/10">
                  <tr>
                    <td className="p-4 font-mono text-alert">49</td>
                    <td className="p-4">Critical Bradycardia</td>
                    <td className="p-4 text-alert font-medium">Emergency</td>
                  </tr>
                  <tr>
                    <td className="p-4 font-mono text-success">50</td>
                    <td className="p-4">Low Normal</td>
                    <td className="p-4 text-success font-medium">Normal</td>
                  </tr>
                  <tr>
                    <td className="p-4 font-mono text-success">120</td>
                    <td className="p-4">High Normal</td>
                    <td className="p-4 text-success font-medium">Normal</td>
                  </tr>
                  <tr>
                    <td className="p-4 font-mono text-alert">121</td>
                    <td className="p-4">Mild Tachycardia</td>
                    <td className="p-4 text-alert font-medium">Emergency</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="text-sm mt-4">
              <em>Note: A secondary Random Forest model, trained on the UCI Heart Disease Dataset (13 clinical features), acts as a background cardiovascular risk assessment tool.</em>
            </p>
          </div>
        </section>

        {/* Decision Logic Section */}
        <section>
          <h2 className="text-2xl font-bold mb-6 text-white border-b border-white/10 pb-4">5. Central Decision & Alert Logic</h2>
          <div className="prose prose-invert max-w-none text-white/70 space-y-4">
            <p>
              The true novelty of the proposed methodology lies in the integration of these predictive models into a cohesive, life-saving physical response, governed by a strict Boolean logic flowchart:
            </p>
            <div className="overflow-x-auto mt-6">
              <table className="w-full text-left border-collapse bg-surface rounded-xl overflow-hidden">
                <thead>
                  <tr className="bg-white/5">
                    <th className="p-4 text-white font-medium">CNN Visual State</th>
                    <th className="p-4 text-white font-medium">SVM Health State</th>
                    <th className="p-4 text-white font-medium">Risk Level</th>
                    <th className="p-4 text-white font-medium">System Action / Intervention</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/10">
                  <tr>
                    <td className="p-4">Normal</td>
                    <td className="p-4">Normal</td>
                    <td className="p-4"><span className="px-2 py-1 rounded bg-success/20 text-success text-xs font-bold uppercase">Safe</span></td>
                    <td className="p-4 text-sm">Continuous Background Monitoring.</td>
                  </tr>
                  <tr>
                    <td className="p-4">Distracted/Drowsy</td>
                    <td className="p-4">Normal</td>
                    <td className="p-4"><span className="px-2 py-1 rounded bg-yellow-500/20 text-yellow-500 text-xs font-bold uppercase">Moderate</span></td>
                    <td className="p-4 text-sm">Trigger In-Cabin Auditory Alarms to re-engage driver.</td>
                  </tr>
                  <tr>
                    <td className="p-4">Normal</td>
                    <td className="p-4">Emergency</td>
                    <td className="p-4"><span className="px-2 py-1 rounded bg-alert/20 text-alert text-xs font-bold uppercase">Critical</span></td>
                    <td className="p-4 text-sm">Activate Hazard Lights, Decelerate Vehicle, Send SOS with GPS.</td>
                  </tr>
                  <tr>
                    <td className="p-4">Distracted/Drowsy</td>
                    <td className="p-4">Emergency</td>
                    <td className="p-4"><span className="px-2 py-1 rounded bg-red-600/20 text-red-500 text-xs font-bold uppercase">Catastrophic</span></td>
                    <td className="p-4 text-sm">Immediate multi-tiered SOS dispatch to medical authorities and vehicle shutdown sequence.</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default Docs;
