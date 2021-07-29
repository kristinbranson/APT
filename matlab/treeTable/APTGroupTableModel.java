
import java.util.Vector;
import javax.swing.table.TableModel;
import com.jidesoft.grid.DefaultGroupTableModel;
import com.jidesoft.grid.DefaultGroupRow;

/**
 * @author Allen Lee
 * @version 1.0
 */

public class APTGroupTableModel extends DefaultGroupTableModel {
    public APTGroupTableModel(TableModel tableModel) {
	super(tableModel);
    }

    @Override
    public DefaultGroupRow createGroupRow() {
	return super.createGroupRow();
    }
    
}
