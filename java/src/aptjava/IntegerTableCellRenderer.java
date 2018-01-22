package aptjava;

import javax.swing.table.DefaultTableCellRenderer;

public class IntegerTableCellRenderer extends DefaultTableCellRenderer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public void setValue(Object value) {
		setText( (value==null) ? "" : String.format("%d", ((java.lang.Double) value).intValue()) );			
	}
}
