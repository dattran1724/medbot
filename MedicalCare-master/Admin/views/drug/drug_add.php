<body id="page-top">

  <!-- Page Wrapper -->
  <div id="wrapper">
    <?php require_once('./views/inc/navbar.php'); ?>
    <!-- Content Wrapper -->
    <div id="content-wrapper" class="d-flex flex-column">

      <!-- Main Content -->
      <div id="content">

        <!-- Topbar -->
        <?php require_once('./views/inc/topbar.php'); ?>
        <!-- End of Topbar -->

        <!-- Begin Page Content -->
        <div class="container-fluid">
          <div class="card shadow mb-4">
            <div class="card-header py-3">
              <h6 class="m-0 font-weight-bold text-primary">Add Category</h6>
            </div>
            <div class="card-body">
              <form action="?act=drug&process=store" method="post" enctype="multipart/form-data">
                <div class="form-group mb-3">
                  <span class="">Name:</span>
                  <input type="text" name="name" class="form-control" placeholder="Name">
                </div>
                <div class="form-group mb-3">
                  <span class="">Description:</span>
                  <textarea name="description" id="descriptionCK"></textarea>
                </div>
                <div class="form-group mb-3">
                  <span class="">Category ID:</span>
                  <select class="form-select mb-3" name="category_id">
                    <?php foreach ($categories as $category) : ?>
                      <option value="<?php echo $category['CategoryID'] ?>"><?php echo $category['CategoryName'] ?></option>
                    <?php endforeach; ?>
                  </select>
                </div>
                <div class="form-group mb-3">
                  <span class="">Information:</span>
                  <textarea name="information" id="informationCK"></textarea>
                </div>
                <div class="form-group mb-3">
                  <span class="">Ingredient:</span>
                  <textarea name="ingredient" id="ingredientCK"></textarea>
                </div>
                <div class="form-group mb-3">
                  <span class="">Instruction:</span>
                  <textarea name="instruction" id="instructionCK"></textarea>
                </div>
                <div class="form-group mb-3">
                  <span class="">Attention:</span>
                  <textarea name="attention" id="attentionCK"></textarea>
                </div>
                <div class="form-group mb-3">
                  <span class="">Price:</span>
                  <input type="text" name="price" class="form-control" placeholder="Price">
                </div>
                <div class="form-group mb-3">
                  <span class="">Quantity:</span>
                  <input type="text" name="quantity" class="form-control" placeholder="Quantity">
                </div>
                <div class="form-group mb-3">
                  <span>Image:</span>
                  <input type="file" class="form-control" name="image">
                </div>
                <div class="form-group mb-3 text-center">
                  <button class="btn btn-lg btn-primary" type="submit">Add</button>
                </div>
              </form>
            </div>
          </div>
        </div>
        <!-- /.container-fluid -->

      </div>
      <!-- End of Main Content -->

</body>
<script>
  ClassicEditor
    .create(document.querySelector('#descriptionCK'))
    .catch(error => {
      console.error(error);
    });
  ClassicEditor
    .create(document.querySelector('#informationCK'))
    .catch(error => {
      console.error(error);
    });
  ClassicEditor
    .create(document.querySelector('#ingredientCK'))
    .catch(error => {
      console.error(error);
    });
  ClassicEditor
    .create(document.querySelector('#instructionCK'))
    .catch(error => {
      console.error(error);
    });
  ClassicEditor
    .create(document.querySelector('#attentionCK'))
    .catch(error => {
      console.error(error);
    });
</script>