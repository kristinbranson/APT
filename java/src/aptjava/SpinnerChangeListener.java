package aptjava;

import java.lang.String;
import javax.swing.JSpinner;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.util.EventObject;

public class SpinnerChangeListener implements ChangeListener  {

	 public void stateChanged(ChangeEvent e) {
     JSpinner s = (JSpinner)(e.getSource());
     //System.out.println("The spinner name is: " + s.getName());
     //System.out.println("The spinner value is: " + s.getValue());
     notifySpinner(s.getName(),((Number) s.getValue()).doubleValue());
   }
	 
	  private java.util.Vector listenerVec = new java.util.Vector();
		public synchronized void addSpinnerListener(SpinnerListener lis) {
			listenerVec.addElement(lis);
		}
		public synchronized void removeSpinnerListener(SpinnerListener lis) {
			listenerVec.removeElement(lis);
		}
		public interface SpinnerListener extends java.util.EventListener {
			void eventCbk(SpinnerEvent event);
		}
		public class SpinnerEvent extends java.util.EventObject {
			private static final long serialVersionUID = 1L;
			public String spinnerName;
			public double spinnerValue;
			SpinnerEvent(Object obj, String name, double val) {
				super(obj);
				this.spinnerName = name;
				this.spinnerValue = val;
			}
		}
		public void notifySpinner(String name,double val) {
			java.util.Vector dataCopy;
			synchronized(this) {
				dataCopy = (java.util.Vector)listenerVec.clone();
			}
			for (int i=0; i < dataCopy.size(); i++) {
				SpinnerEvent event = new SpinnerEvent(this,name,val);
				((SpinnerListener)dataCopy.elementAt(i)).eventCbk(event);
			}
		}
	
}
