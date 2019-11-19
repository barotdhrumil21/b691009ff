var autoNumFormatter = function(cell){
    var row = cell.getRow()
    var rowIndex = row.getPosition(false);
    return (rowIndex+1);
};

var table = new Tabulator("#example-table", {
    reactiveData:true, //enable reactive data
    pagination:"local",
    paginationAddRow:"table",
    layout:"fitColumns",
    history:true,
    height:"400px",
    columns:[
        {title:"Question_no", field:"no",validator:"required",formatter:autoNumFormatter, sorter:"number", width:100, editor:false, htmlOutput:true},
        {title:"Ans", field:"ans",validator:["regex:^[A-E]+[,]*[A-E]*$","maxLength:5"], editor:"input",align:"right", htmlOutput:true},
        {title:"Correct_Marks",validator:["required","max:10"],editor:"number", field:"cans",align:"center", width:100, htmlOutput:true,editorParams:{
            min:0,
            max:10,
            step:10,
            elementAttributes:{
                maxlength:"2", //set the maximum character length of the input element to 10 characters
            }
        }},
        {title:"Wrong_Marks",editor:"number",validator:["required","max:10"], field:"wans", sorter:"number", htmlOutput:true},
        {title:"Subject",validator:["required","regex:^[A-Z]*[a-z]*$"], editor:"input",field:"sub", sorter:"string", align:"center", htmlOutput:true},
        { title:"Remove", formatter:"buttonCross", width:40, align:"center", cellClick:function(e, cell){
            cell.getRow().delete();
            table.redraw(true);
        }},
    ],
});


$("#multiple-add").click(function(){
var no = document.getElementById("rownos").value;
var cmarks = document.getElementById("cmarks").value;
var wmarks = document.getElementById("wmarks").value;
var subject = document.getElementById("subject").value;
for(i=1; i<=no; i++){
    table.addRow({ans:"A",cans:cmarks,wans:wmarks,sub:subject});

}
});

$(window).resize(function(){
    $("#example-table").tabulator("redraw");
  });


  $("#download").click(function(){
    table.download("csv", "data.csv", {bom:true}); //include BOM in output
});

$("#empty").click(function(){
    table.clearData()
});
