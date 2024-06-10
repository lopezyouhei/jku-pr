$categories_list = @('dog_15_categories', $null, 'main_5_categories', 'wen_10_categories')

foreach ($category in $categories_list) {
    if ($null -eq $category) {
        python .\PR\metrics\knn_main.py
    } else {
        python .\PR\metrics\knn_main.py --categories $category
    }
}