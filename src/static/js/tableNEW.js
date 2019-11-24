  var ii = 0
  // console.log($('#ansDATA').text());
  // answer_str = $('#ansDATA').text()
  // new_ans = answer_str.replace(/'/g, '"')
  // var answers = JSON.parse(new_ans)
  // var temp
  // var check = JSON.stringify(answers)
  //
  // console.log("temp",temp);
  // console.log("check",check);

  var ttt = []
  answer_str = $('#ansDATA').text()
  new_ans = answer_str.replace(/'/g, '"')
  //var answers = JSON.parse(new_ans)
  localStorage.setItem("answers", new_ans);

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
              ii = ii-1;
              console.log("after remove:",ii);
          }},
      ],
  });

var table_data = []


  $('#addANS').click(function(){
    setTimeout(function(){
      temp_answers = localStorage.getItem("answers");
      temp_answers= JSON.parse(temp_answers)

      for(i=0; i<100; i++){
        var k=i+1
        var key = 'q'+ k
         ttt.push({ans:temp_answers[key]})
         table_data[i]["ans"]= temp_answers[key]
    }
      table.replaceData(table_data);
      table.redraw(true);

      ttt = [];
      answer_str = "";
      temp_answers = "";

       }, 2000);
       // console.log(table_data);
       // console.log("after loop ",xdata);

  })


  $("#multiple-add").click(function(){

    // console.log("after click:",ii);
  var no = document.getElementById("rownos").value;
  var cmarks = document.getElementById("cmarks").value;
  var wmarks = document.getElementById("wmarks").value;
  var subject = document.getElementById("subject").value;

  var counter = Number(ii) + Number(no)

  if (no <=100 && counter <= 100) {
    for(i=1; i<=no; i++){
        table.addRow({ans:"A",cans:cmarks,wans:wmarks,sub:subject});
        ii = ii+1;
    }

    table_data = table.getData();

    if (no == 0){
      rows = document.getElementById("input_data")
      rows.innerHTML = " no rows added,select more than 0 rows to add "
    }
    else {
      rows = document.getElementById("input_data")
      rows.innerHTML = " " + no + " rows added ";
    }

    document.getElementById("rowadd").classList.add('show')
    setTimeout(function(){
               document.getElementById("rowadd").classList.remove('show');
       }, 2000);

  }
  else {
    rows = document.getElementById("input_data")
    rows.innerHTML = "can not add more,limit reached (100 rows)  TOTAL rows added :" + Number(ii)
    document.getElementById("rowadd").classList.add('show')
    setTimeout(function(){
               document.getElementById("rowadd").classList.remove('show');
       }, 2000);
  }
  // console.log(table.getRow(1))
  });


  $(window).resize(function(){
      $("#example-table").tabulator("redraw");
    });


    $("#download").click(function(){
      table.download("csv", "data.csv", {bom:true}); //include BOM in output
  });

  $("#empty").click(function(){
      table.clearData();
      table_data = [];
      ii = 0;
  });