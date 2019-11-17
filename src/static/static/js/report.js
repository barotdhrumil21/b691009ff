$(document).ready(function(){
 $('#load_data').click(function(){
  $.ajax({
   url:document.getElementById("data").value,
   dataType:"text",
   success:function(data)
   {
    var data_rows = data.split(/\r?\n|\r/);
    var table_data = '<table data-show-export="true" data-toggle="table" class="css-serial table-dark table table-bordered table-responsive-md table-striped text-center">';
    for(var count = 0; count<data_rows.length; count++)
    {
     var cell_data = data_rows[count].split(",");
     table_data += '<tr>';
     for(var cell_count=0; cell_count<cell_data.length; cell_count++)
     {
      if(count === 0)
      {
       table_data += '<th>'+cell_data[cell_count]+'</th>';
      }
      else
      {
       table_data += '<td>'+cell_data[cell_count]+'</td>';
      }
     }
     table_data += '</tr>';
    }
    table_data += '</table>';
    $('#employee_table').html(table_data);
   }
  });
 });

});
