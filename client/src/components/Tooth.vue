<template>
    <div 
        v-bind:style='{ backgroundColor, opacity }'
        class='tooth'>
    </div>
</template>

<script>
    export default {
        name: 'Tooth',
        props: {
            backgroundColor: String,
            opacity: Number,
        }
    }
</script>

<style scoped>
    .tooth {
        border-radius: 100px;
        box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);
        height: 1200px;
        transform: rotate(45deg);
        width: 1200px;
    }
</style>