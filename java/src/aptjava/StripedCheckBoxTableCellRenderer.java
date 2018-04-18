package aptjava;

import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.JCheckBox;

public class StripedCheckBoxTableCellRenderer extends DefaultTableCellRenderer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	private static final java.awt.Color DARKSTRIPE = new java.awt.Color(0.314f,0.314f,0.314f);
  private static final java.awt.Color LITESTRIPE = new java.awt.Color(0.502f,0.502f,0.502f);
  private static final java.awt.Color SELECTED = new java.awt.Color(0.0f,0.482f,0.655f);
  
  private final JCheckBox ckb = new JCheckBox();

  public StripedCheckBoxTableCellRenderer() {
    setOpaque(true);
  }
  
  public java.awt.Component getTableCellRendererComponent(javax.swing.JTable table, Object value, boolean isSelected, boolean hasFocus, int row, int column)
  {
    javax.swing.JComponent c = (javax.swing.JComponent)super.getTableCellRendererComponent(table, value, isSelected, hasFocus, row, column);
    
    if (isSelected) {
    	c.setBackground(SELECTED);
    } else { 
      if (row % 2 == 0) {
        c.setBackground(DARKSTRIPE);
      } else {
        c.setBackground(LITESTRIPE);
      }
    }
    
    if (value instanceof Boolean) {
      ckb.setSelected(((Boolean) value));
      ckb.setHorizontalAlignment(javax.swing.JLabel.CENTER);
      ckb.setBackground(super.getBackground());
      if (isSelected || hasFocus) {
          ckb.setBackground(SELECTED);
      }
      return ckb;
    }
    
    return c;
  }
  
}
